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

// Hyperlight’s x86 guest uses I/O port OUT; s390x uses a `DIAG` with a private
// function code that Linux KVM does not handle in-kernel, yielding
// `KVM_EXIT_S390_SIEIC` to userspace (see `hyperlight-host` `kvm/s390x.rs`).

use core::arch::asm;

use hyperlight_common::outb::S390X_HYPERLIGHT_DIAG_IO;

/// Deliver a 32-bit value to the hypervisor on the logical “port” used for guest→host messages.
///
/// Encoding matches the host decoder in `hyperlight-host` `kvm/s390x.rs`: `DIAG` RS form with
/// diagnose code `S390X_HYPERLIGHT_DIAG_IO`, **port in `GR2`**, **value in `GR3`** (low 32 bits).
///
/// We pin `r2`/`r3` explicitly so the instruction’s `ipa` field always matches what the host
/// decodes from `((ipa >> 4) & 0xf)` / `(ipa & 0xf)`. A free-register `in(reg)` choice can still
/// produce a valid `DIAG`, but if userspace’s `ipa`/`ipb` view and LLVM’s allocation ever disagree
/// with `KVM_GET_REGS`, the host could mis-read the logical port (e.g. treat a host-call as
/// `OutBAction::CallFunction`) while the guest pushed payload for a different outb — surfacing as
/// an empty output buffer (`stack pointer: 8`).
///
/// The guest must run with a PSW that allows `DIAG` (supervisor state; same default as the rest
/// of the s390x Hyperlight bring-up).
pub(crate) unsafe fn out32(port: u16, val: u32) {
    let p = port as u64;
    let v = val as u64;
    unsafe {
        // Unnamed `in("r2")` / `in("r3")`: Rust forbids `name = in("r2") ...` (explicit regs
        // cannot have operand names); use `%r2`/`%r3` in the template, not `{name}`.
        asm!(
            "diag %r2, %r3, {fc}",
            in("r2") p,
            in("r3") v,
            fc = const S390X_HYPERLIGHT_DIAG_IO,
            options(nostack),
        );
    }
}
