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
/// Encoding matches the host decoder: `DIAG R1,R3,I2` with `I2 = S390X_HYPERLIGHT_DIAG_IO`,
/// port in `GR[R1]`, value in `GR[R3]` (low 32 bits). The guest must run with a PSW that
/// allows `DIAG` (supervisor state; same default as the rest of the s390x Hyperlight bring-up).
pub(crate) unsafe fn out32(port: u16, val: u32) {
    let p = port as u64;
    let v = val as u64;
    unsafe {
        asm!(
            "diag {p},{v},{fc}",
            p = in(reg) p,
            v = in(reg) v,
            fc = const S390X_HYPERLIGHT_DIAG_IO,
            options(nostack),
        );
    }
}
