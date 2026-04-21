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

// s390x guest: `dispatch_function` calls Rust `internal_dispatch_function`, then halts via the
// same Hyperlight `DIAG` channel as `out32` (operands on GR4/GR5 so GR2 keeps the `generic_init`
// return / dispatch address for the host). We intentionally omit amd64’s TLB-flush prelude
// (`PTLB` + PSW CC): KVM s390x does not load the snapshot page tables, and `PTLB` prevented
// reaching dispatch in testing (host saw an empty output buffer).

pub(crate) mod context;
pub(crate) mod exception;
pub(crate) mod machine;

use hyperlight_common::outb::{S390X_HYPERLIGHT_DIAG_IO, VmAction};

/// Exit to the host with `VmAction::Halt` (amd64 OUT+hlt equivalent): used after init and dispatch.
///
/// Uses **`%r4` / `%r5`** for the `DIAG` operands (not **`%r2` / `%r3`**): `generic_init` returns the
/// dispatch entry in **`r2`** per the s390x ELF ABI, and `hyperlight_host` `initialise` reads that
/// register to set `NextAction::Call`. `out32` guest→host I/O still pins **`%r2` / `%r3`**.
#[unsafe(no_mangle)]
unsafe extern "C" fn s390x_guest_vm_halt() -> ! {
    let p = VmAction::Halt as u64;
    let v = 0u64;
    unsafe {
        core::arch::asm!(
            "diag %r4, %r5, {fc}",
            in("r4") p,
            in("r5") v,
            fc = const S390X_HYPERLIGHT_DIAG_IO,
            options(nostack),
        );
    }
    loop {
        core::hint::spin_loop();
    }
}

core::arch::global_asm!(
    ".global dispatch_function",
    "dispatch_function:",
    "    brasl %r14, {internal}",
    "    brasl %r14, {halt}",
    internal = sym crate::guest_function::call::internal_dispatch_function,
    halt = sym s390x_guest_vm_halt,
);

pub mod dispatch {
    unsafe extern "C" {
        pub(crate) unsafe fn dispatch_function();
    }
}

/// Guest entry — same parameters as amd64 (`PEB`, RNG seed, guest page size, log filter).
///
/// The host reads **`GR2`** after the first halt to set `NextAction::Call` to the dispatch entry.
/// Keep that value live in **`GR2`** across the halt `DIAG` by constraining it in **`asm!`**
/// (`inlateout("r2")`); a plain Rust call to [`s390x_guest_vm_halt`] can let LLVM spill the return
/// of `generic_init` before the hypercall.
#[unsafe(no_mangle)]
pub extern "C" fn entrypoint(peb_address: u64, seed: u64, ops: u64, max_log_level: u64) -> ! {
    // `inlateout("r2")` requires `mut`: the operand is written back by the assembler.
    let mut dispatch = crate::generic_init(peb_address, seed, ops, max_log_level);
    unsafe {
        core::arch::asm!(
            "diag %r4, %r5, {fc}",
            in("r4") VmAction::Halt as u64,
            in("r5") 0u64,
            inlateout("r2") dispatch,
            fc = const S390X_HYPERLIGHT_DIAG_IO,
            options(nostack),
        );
        loop {
            core::hint::spin_loop();
        }
    }
}
