/*
Copyright 2025  The Hyperlight Authors.

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

use std::sync::LazyLock;
use std::sync::Mutex;

use hyperlight_common::outb::VmAction;
use kvm_bindings::{kvm_regs, kvm_userspace_memory_region};
use kvm_ioctls::Cap::UserMemory;
use kvm_ioctls::{Error as KvmErrno, Kvm, VcpuExit, VcpuFd, VmFd};
use tracing::{Span, instrument};
#[cfg(feature = "trace_guest")]
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::hypervisor::regs::{
    CommonDebugRegs, CommonFpu, CommonRegisters, CommonSpecialRegisters,
};
#[cfg(all(test, not(feature = "nanvix-unstable")))]
use crate::hypervisor::virtual_machine::XSAVE_BUFFER_SIZE;
use crate::hypervisor::virtual_machine::{
    CreateVmError, MapMemoryError, RegisterError, RunVcpuError, UnmapMemoryError, VirtualMachine,
    VmExit,
};
use crate::mem::memory_region::MemoryRegion;
#[cfg(feature = "trace_guest")]
use crate::sandbox::trace::TraceContext as SandboxTraceContext;

/// Default PSW mask for a runnable 64-bit guest (DAT on, wait off). The exact
/// mask is refined when the s390x guest ABI is fully defined.
const DEFAULT_S390_PSW_MASK: u64 = 0x0000_0001_8000_0000;

/// Return `true` if KVM is available, API version is 12, and `KVM_CAP_USER_MEMORY` is present.
#[instrument(skip_all, parent = Span::current(), level = "Trace")]
pub(crate) fn is_hypervisor_present() -> bool {
    if let Ok(kvm) = Kvm::new() {
        let api_version = kvm.get_api_version();
        match api_version {
            version if version == 12 && kvm.check_extension(UserMemory) => true,
            12 => {
                tracing::info!("KVM does not have KVM_CAP_USER_MEMORY capability");
                false
            }
            version => {
                tracing::info!("KVM GET_API_VERSION returned {}, expected 12", version);
                false
            }
        }
    } else {
        tracing::info!("KVM is not available on this system");
        false
    }
}

/// KVM-backed single-vCPU VM for s390x.
#[derive(Debug)]
pub(crate) struct KvmVm {
    vm_fd: VmFd,
    vcpu_fd: VcpuFd,
    /// PSW (addr, mask): applied to `kvm_run` before each `KVM_RUN` because
    /// `VirtualMachine::set_regs` is `&self` and PSW lives in `kvm_run`.
    shadow_psw: Mutex<(u64, u64)>,
}

static KVM: LazyLock<std::result::Result<Kvm, CreateVmError>> =
    LazyLock::new(|| Kvm::new().map_err(|e| CreateVmError::HypervisorNotAvailable(e.into())));

impl KvmVm {
    #[instrument(err(Debug), skip_all, parent = Span::current(), level = "Trace")]
    pub(crate) fn new() -> std::result::Result<Self, CreateVmError> {
        let hv = KVM.as_ref().map_err(|e| e.clone())?;

        let vm_fd = hv
            .create_vm_with_type(0)
            .map_err(|e| CreateVmError::CreateVmFd(e.into()))?;

        let mut vcpu_fd = vm_fd
            .create_vcpu(0)
            .map_err(|e| CreateVmError::CreateVcpuFd(e.into()))?;

        let (init_addr, init_mask) = {
            let run = vcpu_fd.get_kvm_run();
            (run.psw_addr, run.psw_mask)
        };
        let init_mask = if init_mask == 0 {
            DEFAULT_S390_PSW_MASK
        } else {
            init_mask
        };

        Ok(Self {
            vm_fd,
            vcpu_fd,
            shadow_psw: Mutex::new((init_addr, init_mask)),
        })
    }

    fn apply_shadow_psw_to_run(&mut self) {
        let (addr, mask) = *self.shadow_psw.lock().unwrap();
        let run = self.vcpu_fd.get_kvm_run();
        run.psw_addr = addr;
        run.psw_mask = mask;
    }

    fn refresh_shadow_psw_from_run(&mut self) {
        let run = self.vcpu_fd.get_kvm_run();
        let mut g = self.shadow_psw.lock().unwrap();
        g.0 = run.psw_addr;
        g.1 = run.psw_mask;
    }

    fn run_vcpu_default(&mut self) -> std::result::Result<VmExit, RunVcpuError> {
        /// Owned copy of exit data so we can update `kvm_run` (PSW) after `KVM_RUN` returns.
        enum RunExit {
            Halt,
            IoOut(u16, Vec<u8>),
            MmioRead(u64),
            MmioWrite(u64),
            #[cfg(gdb)]
            Debug,
            Unknown(String),
            KernelErr(KvmErrno),
        }

        self.apply_shadow_psw_to_run();
        let mapped = match self.vcpu_fd.run() {
            Ok(VcpuExit::Hlt) => RunExit::Halt,
            Ok(VcpuExit::IoOut(port, data)) => {
                if port == VmAction::Halt as u16 {
                    RunExit::Halt
                } else {
                    RunExit::IoOut(port, data.to_vec())
                }
            }
            Ok(VcpuExit::MmioRead(addr, _)) => RunExit::MmioRead(addr),
            Ok(VcpuExit::MmioWrite(addr, _)) => RunExit::MmioWrite(addr),
            #[cfg(gdb)]
            Ok(VcpuExit::Debug(_)) => RunExit::Debug,
            Ok(other) => RunExit::Unknown(format!("Unknown KVM VCPU exit: {:?}", other)),
            Err(e) => RunExit::KernelErr(e),
        };
        self.refresh_shadow_psw_from_run();

        match mapped {
            RunExit::Halt => Ok(VmExit::Halt()),
            RunExit::IoOut(port, data) => Ok(VmExit::IoOut(port, data)),
            RunExit::MmioRead(addr) => Ok(VmExit::MmioRead(addr)),
            RunExit::MmioWrite(addr) => Ok(VmExit::MmioWrite(addr)),
            #[cfg(gdb)]
            RunExit::Debug => Ok(VmExit::Debug {}),
            RunExit::Unknown(s) => Ok(VmExit::Unknown(s)),
            RunExit::KernelErr(e) => match e.errno() {
                libc::EINTR => Ok(VmExit::Cancelled()),
                libc::EAGAIN => Ok(VmExit::Retry()),
                _ => Err(RunVcpuError::Unknown(e.into())),
            },
        }
    }
}

impl VirtualMachine for KvmVm {
    unsafe fn map_memory(
        &mut self,
        (slot, region): (u32, &MemoryRegion),
    ) -> std::result::Result<(), MapMemoryError> {
        let mut kvm_region: kvm_userspace_memory_region = region.into();
        kvm_region.slot = slot;
        unsafe { self.vm_fd.set_user_memory_region(kvm_region) }
            .map_err(|e| MapMemoryError::Hypervisor(e.into()))
    }

    fn unmap_memory(
        &mut self,
        (slot, region): (u32, &MemoryRegion),
    ) -> std::result::Result<(), UnmapMemoryError> {
        let mut kvm_region: kvm_userspace_memory_region = region.into();
        kvm_region.slot = slot;
        kvm_region.memory_size = 0;
        unsafe { self.vm_fd.set_user_memory_region(kvm_region) }
            .map_err(|e| UnmapMemoryError::Hypervisor(e.into()))
    }

    fn run_vcpu(
        &mut self,
        #[cfg(feature = "trace_guest")] tc: &mut SandboxTraceContext,
    ) -> std::result::Result<VmExit, RunVcpuError> {
        #[cfg(feature = "trace_guest")]
        tc.setup_guest_trace(Span::current().context());

        // `hw-interrupts` is only implemented for x86 KVM; on s390x use the default run path.
        self.run_vcpu_default()
    }

    fn regs(&self) -> std::result::Result<CommonRegisters, RegisterError> {
        let kvm_regs = self
            .vcpu_fd
            .get_regs()
            .map_err(|e| RegisterError::GetRegs(e.into()))?;
        let mut r = CommonRegisters::from(&kvm_regs);
        let (addr, mask) = *self.shadow_psw.lock().unwrap();
        r.rip = addr;
        r.rflags = mask;
        Ok(r)
    }

    fn set_regs(&self, regs: &CommonRegisters) -> std::result::Result<(), RegisterError> {
        let kvm_regs: kvm_regs = regs.into();
        self.vcpu_fd
            .set_regs(&kvm_regs)
            .map_err(|e| RegisterError::SetRegs(e.into()))?;
        let mut g = self.shadow_psw.lock().unwrap();
        g.0 = regs.rip;
        if regs.rflags != 0 {
            g.1 = regs.rflags;
        }
        Ok(())
    }

    fn fpu(&self) -> std::result::Result<CommonFpu, RegisterError> {
        Ok(CommonFpu::default())
    }

    fn set_fpu(&self, _fpu: &CommonFpu) -> std::result::Result<(), RegisterError> {
        Ok(())
    }

    fn sregs(&self) -> std::result::Result<CommonSpecialRegisters, RegisterError> {
        Ok(CommonSpecialRegisters::default())
    }

    fn set_sregs(&self, _sregs: &CommonSpecialRegisters) -> std::result::Result<(), RegisterError> {
        Ok(())
    }

    fn debug_regs(&self) -> std::result::Result<CommonDebugRegs, RegisterError> {
        Ok(CommonDebugRegs::default())
    }

    fn set_debug_regs(&self, _drs: &CommonDebugRegs) -> std::result::Result<(), RegisterError> {
        Ok(())
    }

    fn xsave(&self) -> std::result::Result<Vec<u8>, RegisterError> {
        Ok(vec![0u8; 4096])
    }

    fn reset_xsave(&self) -> std::result::Result<(), RegisterError> {
        Ok(())
    }

    #[cfg(test)]
    #[cfg(not(feature = "nanvix-unstable"))]
    fn set_xsave(&self, xsave: &[u32]) -> std::result::Result<(), RegisterError> {
        if std::mem::size_of_val(xsave) != XSAVE_BUFFER_SIZE {
            return Err(RegisterError::XsaveSizeMismatch {
                expected: XSAVE_BUFFER_SIZE as u32,
                actual: std::mem::size_of_val(xsave) as u32,
            });
        }
        Ok(())
    }
}
