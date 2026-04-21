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
/// Default runnable-guest PSW mask; must stay in sync with `kvm/s390x.rs` `DEFAULT_S390_PSW_MASK`.
const S390_DEFAULT_PSW_MASK: u64 = 0x0000_0001_8000_0000;

#[inline]
fn s390_psw_mask_for_dispatch(base_mask: u64, pending_tlb_flush: bool) -> u64 {
    // x86 uses ZF=1 to request a flush before dispatch; on s390x we use PSW CC=3 (non-zero).
    let cc: u64 = if pending_tlb_flush { 3 } else { 0 };
    (base_mask & !PSW_MASK_CC) | (cc << 44)
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
            // Linux `struct lowcore` `program_new_psw` @ real 0x1d0. All-zero lets KVM / the CPU
            // load an invalid PSW on program-interrupt presentation (e.g. after userspace
            // `KVM_S390_INTERRUPT`), which can recurse into `ICPT_OPEREXC` and spin the host run
            // loop. Prime a disabled-wait PSW (big-endian doublewords: mask, IA). That state
            // surfaces as `KVM_EXIT_S390_SIEIC` / `ICPT_WAIT` (`0x1c`); `kvm/s390x.rs` maps it to
            // `VmExit::Halt` so the run loop can finish.
            const LC_PROGRAM_NEW_PSW: usize = 0x1d0;
            const PSW_MASK_WAIT: u64 = 0x0002_0000_0000_0000;
            const PSW_MASK_EA: u64 = 0x0000_0001_0000_0000;
            const PSW_MASK_BA: u64 = 0x0000_0000_8000_0000;
            let wait_mask: u64 = PSW_MASK_WAIT | PSW_MASK_EA | PSW_MASK_BA;
            let mut psw_lc: [u8; 16] = [0; 16];
            psw_lc[0..8].copy_from_slice(&wait_mask.to_be_bytes());
            psw_lc[8..16].copy_from_slice(&0u64.to_be_bytes());
            low_eshm
                .copy_from_slice(&psw_lc, LC_PROGRAM_NEW_PSW)
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
        guest_max_log_level: Option<LevelFilter>,
        #[cfg(gdb)] dbg_mem_access_fn: Arc<Mutex<SandboxMemoryManager<HostSharedMemory>>>,
    ) -> std::result::Result<(), InitializeError> {
        let NextAction::Initialise(initialise) = self.entrypoint else {
            return Ok(());
        };

        // s390x ELF ABI: integer args in r2–r5 (`kvm_regs::gprs[2..5]` == rcx, rdx, rsi, rdi in
        // `CommonRegisters`), stack in r15, return values in r2 (== `regs.rcx`).
        // `rip` / `rflags` are the guest PSW IA and mask via `kvm_run`.
        let regs = CommonRegisters {
            rip: initialise,
            r15: self.rsp_gva,
            rcx: peb_addr.into(),
            rdx: seed,
            rsi: page_size.into(),
            rdi: super::get_guest_log_filter(guest_max_log_level),
            // `CommonRegisters::rsp` maps to KVM GPR6. The s390x entry ABI passes a fifth
            // argument to `entrypoint` / `generic_init` in GR6: live scratch size in bytes
            // (must match `SandboxMemoryLayout::get_scratch_size` and the scratch memslot).
            rsp: mem_mgr.layout.get_scratch_size() as u64,
            // Non-zero PSW mask: `KvmVm::set_regs` only replaces the shadow mask when `rflags` is
            // set; use the same EA|BA runnable default as dispatch (DAT cleared in the KVM layer).
            rflags: S390_DEFAULT_PSW_MASK,
            ..Default::default()
        };
        self.vm.set_regs(&regs)?;

        self.run(
            mem_mgr,
            host_funcs,
            #[cfg(gdb)]
            dbg_mem_access_fn,
        )
        .map_err(InitializeError::Run)?;

        let regs = self.vm.regs()?;
        let sp = regs.r15;
        if !sp.is_multiple_of(8) {
            return Err(InitializeError::InvalidStackPointer(sp));
        }
        self.rsp_gva = sp;
        self.entrypoint = NextAction::Call(regs.rcx);

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

        let r_current = self.vm.regs().map_err(DispatchGuestCallError::SetupRegs)?;
        let base_psw_mask = if r_current.rflags != 0 {
            r_current.rflags
        } else {
            S390_DEFAULT_PSW_MASK
        };
        let rflags = s390_psw_mask_for_dispatch(base_psw_mask, self.pending_tlb_flush);

        // Keep the full GPR file from the last guest-visible state (e.g. after `generic_init` or
        // a prior halt). Zeroing the rest broke Linux/s390x guests: the ELF ABI uses r12 (TOC) and
        // other callee-saved registers across calls; `KvmVm::regs()` already merges
        // `kvm_run.s.regs` when `KVM_SYNC_GPRS` is set.
        let regs = CommonRegisters {
            rip: dispatch_func_addr,
            r15: self.rsp_gva,
            rflags,
            ..r_current
        };
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
            self.pending_tlb_flush = true;
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
}
