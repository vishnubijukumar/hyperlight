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

// s390x guest bring-up: run the same Rust init path as amd64, without the x86
// asm (GDT/TSS/paging). The host must supply a valid stack pointer (r15) before
// calling `entrypoint`, or this will fault—real KVM setup will set that.
//
// After init / dispatch we spin until the host pre-empts the vCPU. Replace with
// a proper architected wait (e.g. WAIT / DIAG) once the KVM exit model matches.

/// Placeholder for amd64’s OUT+hlt: keep the vCPU parked until the host runs it again.
#[inline(never)]
fn halt_forever() -> ! {
    loop {
        core::hint::spin_loop();
    }
}

pub mod dispatch {
    /// Host invokes this for each guest function call (amd64 wraps this in asm for TLB flush).
    #[unsafe(no_mangle)]
    pub extern "C" fn dispatch_function() {
        crate::guest_function::call::internal_dispatch_function();
        super::halt_forever();
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
    halt_forever();
}
