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

// s390x guest: dispatch matches amd64’s asm stub shape — TLB hint via PSW condition code,
// then halt via the same DIAG channel as `hyperlight_guest::arch::s390x::out32`.

pub(crate) mod context;
pub(crate) mod exception;
pub(crate) mod machine;

use hyperlight_common::outb::{S390X_HYPERLIGHT_DIAG_IO, VmAction};

/// Exit to the host with `VmAction::Halt` (amd64 OUT+hlt equivalent): used after init and dispatch.
#[unsafe(no_mangle)]
unsafe extern "C" fn s390x_guest_vm_halt() -> ! {
    let p = VmAction::Halt as u64;
    let v = 0u64;
    unsafe {
        // Match `hyperlight_guest::arch::s390x::exit::out32`: pin GR2/GR3 so host `ipa`/`KVM_GET_REGS`
        // always agree on the logical port for `kvm/s390x.rs` decode.
        core::arch::asm!(
            "diag %r2, %r3, {fc}",
            in("r2") p,
            in("r3") v,
            fc = const S390X_HYPERLIGHT_DIAG_IO,
            options(nostack),
        );
    }
    loop {
        core::hint::spin_loop();
    }
}

// `brc 8, 1f`: branch if CC == 0 (same sense as amd64 `jnz` over ZF — skip flush when no hint).
core::arch::global_asm!(
    ".global dispatch_function",
    "dispatch_function:",
    "    brc 8, 1f",
    "    ptlb",
    "1:",
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
#[unsafe(no_mangle)]
pub extern "C" fn entrypoint(
    peb_address: u64,
    seed: u64,
    ops: u64,
    max_log_level: u64,
) -> ! {
    let _dispatch_addr = crate::generic_init(peb_address, seed, ops, max_log_level);
    unsafe { s390x_guest_vm_halt() }
}
