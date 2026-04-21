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

//! s390x register views shared with the rest of the host.
//! GPRs map to KVM `kvm_regs::gprs` in the same order as the x86 field layout
//! (rax..r15). `rip` / `rflags` shadow the guest PSW address and mask; they are
//! not part of `KVM_SET_REGS` and are applied via `kvm_run` inside the KVM backend.

#[cfg(kvm)]
use kvm_bindings::kvm_regs;

// Match x86 defaults used by reset paths and tests that compile on every arch.
pub(crate) const FP_CONTROL_WORD_DEFAULT: u16 = 0x37f;
pub(crate) const MXCSR_DEFAULT: u32 = 0x1f80;

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub(crate) struct CommonSegmentRegister {
    pub base: u64,
    pub limit: u32,
    pub selector: u16,
    pub type_: u8,
    pub present: u8,
    pub dpl: u8,
    pub db: u8,
    pub s: u8,
    pub l: u8,
    pub g: u8,
    pub avl: u8,
    pub unusable: u8,
    pub padding: u8,
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub(crate) struct CommonTableRegister {
    pub base: u64,
    pub limit: u16,
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub(crate) struct CommonSpecialRegisters {
    pub cs: CommonSegmentRegister,
    pub ds: CommonSegmentRegister,
    pub es: CommonSegmentRegister,
    pub fs: CommonSegmentRegister,
    pub gs: CommonSegmentRegister,
    pub ss: CommonSegmentRegister,
    pub tr: CommonSegmentRegister,
    pub ldt: CommonSegmentRegister,
    pub gdt: CommonTableRegister,
    pub idt: CommonTableRegister,
    pub cr0: u64,
    pub cr2: u64,
    pub cr3: u64,
    pub cr4: u64,
    pub cr8: u64,
    pub efer: u64,
    pub apic_base: u64,
    pub interrupt_bitmap: [u64; 4],
}

impl CommonSpecialRegisters {
    /// x86 page tables are not used on s390x; callers may still invoke this for a uniform API.
    pub(crate) fn standard_64bit_defaults(_pml4_addr: u64) -> Self {
        Self::default()
    }

    #[cfg(feature = "nanvix-unstable")]
    pub(crate) fn standard_real_mode_defaults() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CommonFpu {
    pub fpr: [[u8; 16]; 8],
    pub fcw: u16,
    pub fsw: u16,
    pub ftwx: u8,
    pub last_opcode: u16,
    pub last_ip: u64,
    pub last_dp: u64,
    pub xmm: [[u8; 16]; 16],
    pub mxcsr: u32,
}

impl Default for CommonFpu {
    fn default() -> Self {
        Self {
            fpr: [[0u8; 16]; 8],
            fcw: FP_CONTROL_WORD_DEFAULT,
            fsw: 0,
            ftwx: 0,
            last_opcode: 0,
            last_ip: 0,
            last_dp: 0,
            xmm: [[0u8; 16]; 16],
            mxcsr: MXCSR_DEFAULT,
        }
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub(crate) struct CommonDebugRegs {
    pub dr0: u64,
    pub dr1: u64,
    pub dr2: u64,
    pub dr3: u64,
    pub dr6: u64,
    pub dr7: u64,
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub(crate) struct CommonRegisters {
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rsp: u64,
    pub rbp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    /// Guest instruction address (KVM `kvm_run.psw_addr`).
    pub rip: u64,
    /// Guest PSW mask (`kvm_run.psw_mask`), not x86 RFLAGS.
    pub rflags: u64,
}

#[cfg(kvm)]
impl From<&kvm_regs> for CommonRegisters {
    fn from(k: &kvm_regs) -> Self {
        let g = &k.gprs;
        Self {
            rax: g[0],
            rbx: g[1],
            rcx: g[2],
            rdx: g[3],
            rsi: g[4],
            rdi: g[5],
            rsp: g[6],
            rbp: g[7],
            r8: g[8],
            r9: g[9],
            r10: g[10],
            r11: g[11],
            r12: g[12],
            r13: g[13],
            r14: g[14],
            r15: g[15],
            rip: 0,
            rflags: 0,
        }
    }
}

#[cfg(kvm)]
impl From<&CommonRegisters> for kvm_regs {
    fn from(regs: &CommonRegisters) -> Self {
        kvm_regs {
            gprs: [
                regs.rax, regs.rbx, regs.rcx, regs.rdx, regs.rsi, regs.rdi, regs.rsp, regs.rbp,
                regs.r8, regs.r9, regs.r10, regs.r11, regs.r12, regs.r13, regs.r14, regs.r15,
            ],
        }
    }
}
