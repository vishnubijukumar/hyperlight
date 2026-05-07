/*
Copyright 2025 The Hyperlight Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

//! IBM z/Architecture (s390x) host: Linux KVM only (MSHV is disabled via `build.rs` cfg).

#[cfg(gdb)]
use std::collections::HashMap;
#[cfg(any(kvm, mshv3))]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU8;
#[cfg(any(kvm, mshv3))]
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use tracing::{Span, instrument};
use tracing_core::LevelFilter;

use super::*;
use crate::hypervisor::InterruptHandleImpl;
#[cfg(any(kvm, mshv3))]
use crate::hypervisor::LinuxInterruptHandle;
#[cfg(gdb)]
use crate::hypervisor::gdb::{
    DebugCommChannel, DebugMsg, DebugResponse, DebuggableVm, VcpuStopReason,
};
#[cfg(gdb)]
use crate::hypervisor::gdb::{DebugError, DebugMemoryAccessError};
use crate::hypervisor::regs::{
    CommonDebugRegs, CommonFpu, CommonRegisters, CommonSpecialRegisters,
};
#[cfg(not(gdb))]
use crate::hypervisor::virtual_machine::VirtualMachine;
#[cfg(kvm)]
use crate::hypervisor::virtual_machine::kvm::KvmVm;
#[cfg(mshv3)]
use crate::hypervisor::virtual_machine::mshv::MshvVm;
#[cfg(target_os = "windows")]
use crate::hypervisor::virtual_machine::whp::WhpVm;
use crate::hypervisor::virtual_machine::{
    HypervisorType, RegisterError, VmError, get_available_hypervisor,
};
#[cfg(target_os = "windows")]
use crate::hypervisor::{PartitionState, WindowsInterruptHandle};
use crate::mem::memory_region::MemoryRegionType;
use crate::mem::mgr::SandboxMemoryManager;
use crate::mem::ptr::RawPtr;
use crate::mem::shared_mem::ExclusiveSharedMemory;
use crate::mem::shared_mem::{GuestSharedMemory, HostSharedMemory};
use crate::sandbox::SandboxConfiguration;
use crate::sandbox::host_funcs::FunctionRegistry;
use crate::sandbox::snapshot::NextAction;
#[cfg(feature = "mem_profile")]
use crate::sandbox::trace::MemTraceInfo;
#[cfg(crashdump)]
use crate::sandbox::uninitialized::SandboxRuntimeConfig;

/// Condition-code field in the PSW mask (`PSW_MASK_CC` in Linux `arch/s390/include/uapi/asm/ptrace.h`).
const PSW_MASK_CC: u64 = 0x0000_3000_0000_0000;
/// `PSW_MASK_DAT` from Linux `arch/s390/include/uapi/asm/ptrace.h` (see `kvm/s390x.rs`).
const PSW_MASK_DAT: u64 = 0x0400_0000_0000_0000;
/// `PSW_MASK_PSTATE` — problem state; if set, privileged instructions trap (see `kvm/s390x.rs`).
const PSW_MASK_PSTATE: u64 = 0x0001_0000_0000_0000;
/// `PSW_MASK_WAIT` — see `kvm/s390x.rs` (`psw_mask_hyperlight`).
const PSW_MASK_WAIT: u64 = 0x0002_0000_0000_0000;
/// Default runnable-guest PSW mask; must stay in sync with `kvm/s390x.rs` `DEFAULT_S390_PSW_MASK`.
const S390_DEFAULT_PSW_MASK: u64 = 0x0000_0001_8000_0000;

/// Match `kvm/s390x::psw_mask_hyperlight`: never run the guest dispatch PSW with DAT, PSTATE, or
/// WAIT set.
#[inline]
fn s390_psw_mask_strip_dat_pstate(mask: u64) -> u64 {
    mask & !PSW_MASK_DAT & !PSW_MASK_PSTATE & !PSW_MASK_WAIT
}

#[inline]
fn s390_psw_mask_for_dispatch(base_mask: u64, pending_tlb_flush: bool) -> u64 {
    // x86 uses ZF=1 to request a flush before dispatch; on s390x we use PSW CC=3 (non-zero).
    let base = s390_psw_mask_strip_dat_pstate(if base_mask == 0 {
        S390_DEFAULT_PSW_MASK
    } else {
        base_mask
    });
    let cc: u64 = if pending_tlb_flush { 3 } else { 0 };
    (base & !PSW_MASK_CC) | (cc << 44)
}

impl HyperlightVm {
    /// Create a new HyperlightVm instance (will not run vm until calling `initialise`)
    #[instrument(err(Debug), skip_all, parent = Span::current(), level = "Trace")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        snapshot_mem: SnapshotSharedMemory<GuestSharedMemory>,
        scratch_mem: GuestSharedMemory,
        pml4_addr: u64,
        entrypoint: NextAction,
        rsp_gva: u64,
        page_size: usize,
        #[cfg_attr(target_os = "windows", allow(unused_variables))] config: &SandboxConfiguration,
        #[cfg(gdb)] gdb_conn: Option<DebugCommChannel<DebugResponse, DebugMsg>>,
        #[cfg(crashdump)] rt_cfg: SandboxRuntimeConfig,
        #[cfg(feature = "mem_profile")] trace_info: MemTraceInfo,
    ) -> std::result::Result<Self, CreateHyperlightVmError> {
        #[cfg(gdb)]
        type VmType = Box<dyn DebuggableVm>;
        #[cfg(not(gdb))]
        type VmType = Box<dyn VirtualMachine>;

        let vm: VmType = match get_available_hypervisor() {
            #[cfg(kvm)]
            Some(HypervisorType::Kvm) => Box::new(KvmVm::new().map_err(VmError::CreateVm)?),
            #[cfg(mshv3)]
            Some(HypervisorType::Mshv) => Box::new(MshvVm::new().map_err(VmError::CreateVm)?),
            #[cfg(target_os = "windows")]
            Some(HypervisorType::Whp) => Box::new(WhpVm::new().map_err(VmError::CreateVm)?),
            None => return Err(CreateHyperlightVmError::NoHypervisorFound),
        };

        #[cfg(not(feature = "nanvix-unstable"))]
        vm.set_sregs(&CommonSpecialRegisters::standard_64bit_defaults(pml4_addr))
            .map_err(VmError::Register)?;
        #[cfg(feature = "nanvix-unstable")]
        vm.set_sregs(&CommonSpecialRegisters::standard_real_mode_defaults())
            .map_err(VmError::Register)?;

        #[cfg(any(kvm, mshv3))]
        let interrupt_handle: Arc<dyn InterruptHandleImpl> = Arc::new(LinuxInterruptHandle {
            state: AtomicU8::new(0),
            #[cfg(all(
                target_arch = "x86_64",
                target_vendor = "unknown",
                target_os = "linux",
                target_env = "musl"
            ))]
            tid: AtomicU64::new(unsafe { libc::pthread_self() as u64 }),
            #[cfg(not(all(
                target_arch = "x86_64",
                target_vendor = "unknown",
                target_os = "linux",
                target_env = "musl"
            )))]
            tid: AtomicU64::new(unsafe { libc::pthread_self() }),
            retry_delay: config.get_interrupt_retry_delay(),
            sig_rt_min_offset: config.get_interrupt_vcpu_sigrtmin_offset(),
            dropped: AtomicBool::new(false),
        });

        #[cfg(target_os = "windows")]
        let interrupt_handle: Arc<dyn InterruptHandleImpl> = Arc::new(WindowsInterruptHandle {
            state: AtomicU8::new(0),
            partition_state: std::sync::RwLock::new(PartitionState {
                handle: vm.partition_handle(),
                dropped: false,
            }),
        });

        let snapshot_slot = 0u32;
        let scratch_slot = 1u32;
        #[cfg_attr(not(gdb), allow(unused_mut))]
        let mut ret = Self {
            vm,
            entrypoint,
            rsp_gva,
            interrupt_handle,
            page_size,

            next_slot: scratch_slot + 1,
            freed_slots: Vec::new(),

            snapshot_slot,
            snapshot_memory: None,
            scratch_slot,
            scratch_memory: None,

            s390x_lowcore_guest_mem: None,

            #[cfg(not(feature = "nanvix-unstable"))]
            root_pt_gpa: pml4_addr,

            mmap_regions: Vec::new(),

            pending_tlb_flush: false,

            #[cfg(gdb)]
            gdb_conn,
            #[cfg(gdb)]
            sw_breakpoints: HashMap::new(),
            #[cfg(feature = "mem_profile")]
            trace_info,
            #[cfg(crashdump)]
            rt_cfg,
        };

        // Guest RAM starts at `SandboxMemoryLayout::BASE_ADDRESS` (1 MiB). With KVM initial CPU
        // reset the prefix is 0, so PSA / lowcore live at guest absolute 0..1MiB. That range
        // must be backed or the first `KVM_RUN` can fail with `EFAULT` when the kernel touches it.
        {
            const MIB: usize = 1 << 20;
            let mut low_eshm = ExclusiveSharedMemory::new(MIB)
                .map_err(|e| CreateHyperlightVmError::SharedMemorySetup(e.to_string()))?;
            // Linux s390x `struct lowcore` contains multiple PSWs. If we leave them as all-zero
            // and the guest takes a program interruption early in bring-up, the CPU can load a
            // nonsense PSW and end up "executing" inside lowcore (e.g. at 0x102), leading to an
            // `ICPT_WAIT` / `ICPT_OPEREXC` loop that never advances scratch I/O.
            //
            // Prime:
            // - `restart_psw` @ 0x1a0: point at our current intended entrypoint.
            // - `program_new_psw` @ 0x1d0: point at a tiny lowcore-resident stub that triggers a
            //   Hyperlight `outb Abort` (explicit failure) without requiring scratch to work.
            const LC_STUB: usize = 0x100;
            const LC_RESTART_PSW: usize = 0x1a0;
            const LC_PROGRAM_NEW_PSW: usize = 0x1d0;
            const PSW_MASK_EA: u64 = 0x0000_0001_0000_0000;
            const PSW_MASK_BA: u64 = 0x0000_0000_8000_0000;
            let runnable_mask: u64 = PSW_MASK_EA | PSW_MASK_BA;

            let entry_ia: u64 = match entrypoint {
                NextAction::Initialise(addr) => addr,
                NextAction::Call(addr) => addr,
                #[cfg(test)]
                NextAction::None => 0,
            };

            let mut restart_psw: [u8; 16] = [0; 16];
            restart_psw[0..8].copy_from_slice(&runnable_mask.to_be_bytes());
            restart_psw[8..16].copy_from_slice(&entry_ia.to_be_bytes());
            low_eshm
                .copy_from_slice(&restart_psw, LC_RESTART_PSW)
                .map_err(|e| CreateHyperlightVmError::SharedMemorySetup(format!("{e:#}")))?;

            // Lowcore stub: send an `outb Abort` with code=8 and a terminator (0xFF) in a single
            // DIAG exit, so even if the host doesn't advance PSW for this intercept we won't spam
            // the abort buffer and overflow it.
            //
            // r4 = OutBAction::Abort (102)
            // r5 = u32 packed as little-endian [len, b1, b2, b3]
            //   - [2, 8, 0xFF, 0]  (code byte + terminator)
            // DIAG r4,r5,0x3e8 triggers the host intercept path.
            //
            // Instruction encodings (big-endian bytes):
            // - lghi r4, 102     => a7 49 00 66
            // - lghi r5, 0xff    => a7 59 00 ff
            // - sllg r5,r5,16    => eb 55 00 10 00 0d
            // - oill r5, 0x0802  => a5 5b 08 02
            // - diag r4,r5,0x3e8 => 83 45 03 e8
            // - j .              => a7 f4 00 00  (safety loop)
            let stub: [u8; 26] = [
                0xa7, 0x49, 0x00, 0x66, // lghi r4,102
                0xa7, 0x59, 0x00, 0xff, // lghi r5,255
                0xeb, 0x55, 0x00, 0x10, 0x00, 0x0d, // sllg r5,r5,16
                0xa5, 0x5b, 0x08, 0x02, // oill r5,0x0802
                0x83, 0x45, 0x03, 0xe8, // diag r4,r5,0x3e8
                0xa7, 0xf4, 0x00, 0x00, // j . (spin safely if resumed)
            ];
            low_eshm
                .copy_from_slice(&stub, LC_STUB)
                .map_err(|e| CreateHyperlightVmError::SharedMemorySetup(format!("{e:#}")))?;

            let mut program_new_psw: [u8; 16] = [0; 16];
            program_new_psw[0..8].copy_from_slice(&runnable_mask.to_be_bytes());
            program_new_psw[8..16].copy_from_slice(&(LC_STUB as u64).to_be_bytes());
            low_eshm
                .copy_from_slice(&program_new_psw, LC_PROGRAM_NEW_PSW)
                .map_err(|e| CreateHyperlightVmError::SharedMemorySetup(format!("{e:#}")))?;

            let (_h, low_guest) = low_eshm.build();
            let low_rgn = low_guest.mapping_at(0, MemoryRegionType::S390xLowcore);
            unsafe {
                ret.map_region(&low_rgn)?;
            }
            ret.s390x_lowcore_guest_mem = Some(low_guest);
        }

        ret.update_snapshot_mapping(snapshot_mem)?;
        ret.update_scratch_mapping(scratch_mem)?;

        #[cfg(gdb)]
        if ret.gdb_conn.is_some() {
            ret.send_dbg_msg(DebugResponse::InterruptHandle(ret.interrupt_handle.clone()))?;
            ret.vm.set_debug(true).map_err(VmError::Debug)?;
            if let NextAction::Initialise(initialise) = entrypoint {
                ret.vm
                    .add_hw_breakpoint(initialise)
                    .map_err(CreateHyperlightVmError::AddHwBreakpoint)?;
            }
        }

        Ok(ret)
    }

    /// Initialise the internally stored vCPU with the given PEB address and
    /// random number seed, then run until guest halt (or equivalent exit).
    #[instrument(err(Debug), skip_all, parent = Span::current(), level = "Trace")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn initialise(
        &mut self,
        peb_addr: RawPtr,
        seed: u64,
        page_size: u32,
        mem_mgr: &mut SandboxMemoryManager<HostSharedMemory>,
        host_funcs: &Arc<Mutex<FunctionRegistry>>,
        _guest_max_log_level: Option<LevelFilter>,
        #[cfg(gdb)] dbg_mem_access_fn: Arc<Mutex<SandboxMemoryManager<HostSharedMemory>>>,
    ) -> std::result::Result<(), InitializeError> {
        let NextAction::Initialise(initialise) = self.entrypoint else {
            return Ok(());
        };

        // s390x ELF ABI: integer args in r2–r5 (`kvm_regs::gprs[2..5]` == rcx, rdx, rsi, rdi in
        // `CommonRegisters`), stack in r15, return values in r2 (== `regs.rcx`).
        // `rip` / `rflags` are the guest PSW IA and mask via `kvm_run`.
        let mut regs = CommonRegisters {
            rip: initialise,
            r15: self.rsp_gva,
            rcx: peb_addr.into(),
            rdx: seed,
            rsi: page_size.into(),
            // s390x bring-up: disable guest logging until the KVM run loop no longer hits
            // lowcore disabled-wait paths that make host/guest log handshakes unreliable.
            rdi: 0,
            // `CommonRegisters::rsp` maps to KVM GPR6. The s390x entry ABI passes a fifth
            // argument to `entrypoint` / `generic_init` in GR6: live scratch size in bytes
            // (must match `SandboxMemoryLayout::get_scratch_size` and the scratch memslot).
            rsp: mem_mgr.layout.get_scratch_size() as u64,
            // Non-zero PSW mask: `KvmVm::set_regs` only replaces the shadow mask when `rflags` is
            // set; use the same EA|BA runnable default as dispatch (DAT cleared in the KVM layer).
            rflags: S390_DEFAULT_PSW_MASK,
            ..Default::default()
        };
        // s390x ELF PIC code relies on `%r12` as the GOT/TOC base. If we don't seed it before the
        // guest entrypoint runs, the very first global access can program-interrupt and land in
        // lowcore `program_new_psw` (disabled wait), making the host think the guest halted while
        // scratch I/O cursors never advanced.
        if let Some(got_base) = mem_mgr.guest_s390x_got_base_gva {
            regs.r12 = got_base;
        }
        self.vm.set_regs(&regs)?;

        self.run(
            mem_mgr,
            host_funcs,
            #[cfg(gdb)]
            dbg_mem_access_fn,
        )
        .map_err(InitializeError::Run)?;

        let regs = self.vm.regs()?;
        let dispatch_gva = mem_mgr
            .guest_dispatch_entry_gva
            .unwrap_or(regs.rcx);
        let sp = regs.r15;
        if !sp.is_multiple_of(8) {
            return Err(InitializeError::InvalidStackPointer(sp));
        }
        self.rsp_gva = sp;
        self.entrypoint = NextAction::Call(dispatch_gva);

        Ok(())
    }

    pub(crate) fn get_root_pt(&self) -> Result<u64, AccessPageTableError> {
        #[cfg(feature = "nanvix-unstable")]
        {
            Ok(0)
        }
        #[cfg(not(feature = "nanvix-unstable"))]
        {
            // KVM s390x does not load x86-style page tables; `sregs().cr3` is meaningless (often 0).
            // Snapshot rebuild and other host walks must use the GPA where the guest PT lives
            // (`SandboxMemoryLayout::get_pt_base_gpa`), passed through `HyperlightVm::new` as
            // `pml4_addr`.
            Ok(self.root_pt_gpa)
        }
    }

    pub(crate) fn get_snapshot_sregs(
        &mut self,
    ) -> Result<CommonSpecialRegisters, AccessPageTableError> {
        Ok(self.vm.sregs()?)
    }

    #[instrument(err(Debug), skip_all, parent = Span::current(), level = "Trace")]
    pub(crate) fn dispatch_call_from_host(
        &mut self,
        mem_mgr: &mut SandboxMemoryManager<HostSharedMemory>,
        host_funcs: &Arc<Mutex<FunctionRegistry>>,
        #[cfg(gdb)] dbg_mem_access_fn: Arc<Mutex<SandboxMemoryManager<HostSharedMemory>>>,
    ) -> std::result::Result<(), DispatchGuestCallError> {
        let NextAction::Call(dispatch_func_addr) = self.entrypoint else {
            return Err(DispatchGuestCallError::Uninitialized);
        };

        mem_mgr
            .sync_s390_peb_io_scratch_pointers_from_live_scratch()
            .map_err(|e| {
                DispatchGuestCallError::PreDispatch(format!(
                    "s390x PEB I/O scratch pointer re-sync failed: {e:#}"
                ))
            })?;

        // Order scratch IPC writes (push_buffer) and PEB pointer patches before KVM_RUN so the
        // guest never observes the old stack cursor with the new I/O GPA window (or vice versa).
        #[cfg(not(feature = "nanvix-unstable"))]
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        let r_current = self.vm.regs().map_err(DispatchGuestCallError::SetupRegs)?;
        let base_raw = if r_current.rflags != 0 {
            r_current.rflags
        } else {
            S390_DEFAULT_PSW_MASK
        };
        let base_psw_mask = s390_psw_mask_strip_dat_pstate(base_raw);
        let rflags = s390_psw_mask_for_dispatch(base_psw_mask, self.pending_tlb_flush);

        let mut regs = r_current;
        regs.rip = dispatch_func_addr;
        regs.r15 = self.rsp_gva;
        regs.rflags = rflags;
        regs.rcx = 0;
        regs.rdx = 0;
        regs.rsi = 0;
        regs.rdi = 0;
        if let Some(got_base) = mem_mgr.guest_s390x_got_base_gva {
            regs.r12 = got_base;
        }

        self.vm
            .set_regs(&regs)
            .map_err(DispatchGuestCallError::SetupRegs)?;

        self.vm
            .set_fpu(&CommonFpu::default())
            .map_err(DispatchGuestCallError::SetupRegs)?;

        let result = self
            .run(
                mem_mgr,
                host_funcs,
                #[cfg(gdb)]
                dbg_mem_access_fn,
            )
            .map_err(DispatchGuestCallError::Run);

        self.pending_tlb_flush = false;

        result
    }

    pub(crate) fn reset_vcpu(
        &mut self,
        cr3: u64,
        sregs: &CommonSpecialRegisters,
    ) -> std::result::Result<(), RegisterError> {
        self.vm.set_regs(&CommonRegisters {
            rflags: 0,
            ..Default::default()
        })?;
        self.vm.set_debug_regs(&CommonDebugRegs::default())?;
        self.vm.reset_xsave()?;

        #[cfg(not(feature = "nanvix-unstable"))]
        {
            let mut sregs = *sregs;
            sregs.cr3 = cr3;
            // `VirtualMachine::set_sregs` is a no-op on Linux KVM s390x: there is no CR3 reload.
            // `pending_tlb_flush` exists so the *next* amd64 dispatch sets RFLAGS.ZF and the guest
            // runs a TLB-flush prelude. On s390x the host maps that intent to PSW CC=3
            // (`s390_psw_mask_for_dispatch`), but the guest dispatch stub omits `PTLB` (see
            // `hyperlight_guest_bin` s390x `mod.rs`). Setting `pending_tlb_flush` here would only
            // force CC=3 on the next dispatch with no guest-side handler — leave CC cleared.
            self.pending_tlb_flush = false;
            self.vm.set_sregs(&sregs)?;
        }
        #[cfg(feature = "nanvix-unstable")]
        {
            let _ = (cr3, sregs);
            self.vm
                .set_sregs(&CommonSpecialRegisters::standard_real_mode_defaults())?;
        }

        Ok(())
    }

    /// Dump the first 0x200 bytes of lowcore/PSA (guest real 0) for s390x KVM bring-up.
    ///
    /// This is intended for diagnosing program interruptions that load `program_new_psw` and
    /// end up in disabled-wait (`ICPT_WAIT`, PSW IA=0), which otherwise looks like a normal halt
    /// and leads to the host reading empty scratch I/O (`stack pointer: 8`).
    #[cfg(all(target_os = "linux", target_arch = "s390x", not(feature = "nanvix-unstable")))]
    pub(crate) fn dump_s390x_lowcore(&self) {
        let Some(low) = &self.s390x_lowcore_guest_mem else {
            eprintln!("s390x_lowcore_dump: no lowcore mapping present");
            return;
        };
        let mut buf = [0u8; 0x400];
        if low.copy_to_slice(&mut buf, 0).is_err() {
            eprintln!("s390x_lowcore_dump: copy_to_slice failed");
            return;
        }
        eprintln!("s390x_lowcore_dump[0..0x400]:");
        for (i, chunk) in buf.chunks(16).enumerate() {
            let off = i * 16;
            eprint!("{off:04x}:");
            for b in chunk {
                eprint!(" {b:02x}");
            }
            eprintln!();
        }
        // Also show the exact bytes we primed for `program_new_psw` (0x1d0..0x1df).
        eprint!("s390x_lowcore_dump program_new_psw[0x1d0..0x1e0]:");
        for b in &buf[0x1d0..0x1e0] {
            eprint!(" {b:02x}");
        }
        eprintln!();

        // Decode key lowcore fields (Linux `arch/s390/include/asm/lowcore.h`):
        // - `pgm_int_code`: ILC @ 0x8c (u16), code @ 0x8e (u16)
        // - `data_exc_code`: 0x90 (u32)
        // - `trans_exc_code`: 0x00a8 (u64)
        // - `failing_storage_address`: 0x00f8 (u64)
        // - `pgm_last_break`: 0x110 (u64)
        // - `program_old_psw`: 0x150 (u128 = mask, ia; big-endian doublewords)
        let pgm_ilc = u16::from_be_bytes([buf[0x8c], buf[0x8d]]);
        let pgm_code = u16::from_be_bytes([buf[0x8e], buf[0x8f]]);
        let data_exc_code = u32::from_be_bytes(buf[0x90..0x94].try_into().unwrap());
        let trans_exc_code = u64::from_be_bytes(buf[0x0a8..0x0b0].try_into().unwrap());
        let failing_storage_address = u64::from_be_bytes(buf[0x0f8..0x100].try_into().unwrap());
        let pgm_last_break = u64::from_be_bytes(buf[0x110..0x118].try_into().unwrap());
        let program_old_mask = u64::from_be_bytes(buf[0x150..0x158].try_into().unwrap());
        let program_old_ia = u64::from_be_bytes(buf[0x158..0x160].try_into().unwrap());
        eprintln!(
            "s390x_lowcore_decode: pgm_ilc={:#x} pgm_code={:#x} data_exc_code={:#x} trans_exc_code={:#x} failing_storage_address={:#x} pgm_last_break={:#x} program_old_psw: mask={:#x} ia={:#x}",
            pgm_ilc,
            pgm_code,
            data_exc_code,
            trans_exc_code,
            failing_storage_address,
            pgm_last_break,
            program_old_mask,
            program_old_ia
        );
    }
}
