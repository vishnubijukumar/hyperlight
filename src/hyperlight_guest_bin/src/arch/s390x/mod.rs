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

// Bring-up debug printing uses one DIAG intercept per character and can dominate execution
// during init retries. Keep it off by default so examples like `hello-world` can make progress.
const S390X_BRINGUP_DEBUG_PRINTS: bool = false;

#[inline]
fn s390x_debug_print_bytes(bytes: &[u8]) {
    // DebugPrint path expects UTF-8; our payloads are ASCII.
    let s = unsafe { core::str::from_utf8_unchecked(bytes) };
    hyperlight_guest::exit::debug_print(s);
}

#[inline]
fn s390x_debug_print_hex_u64(label: &[u8], v: u64) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut buf = [0u8; 2 + 16];
    buf[0] = b'0';
    buf[1] = b'x';
    for i in 0..16 {
        let shift = 60usize.saturating_sub(i * 4);
        let nib = ((v >> shift) & 0xf) as usize;
        buf[2 + i] = HEX[nib];
    }
    s390x_debug_print_bytes(label);
    s390x_debug_print_bytes(&buf);
    s390x_debug_print_bytes(b"\n");
}

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

/// First halt after init: host reads **`%r2`** for `NextAction::Call` (dispatch entry).
///
/// `dispatch` is the **first** parameter with **`extern "C"`** so it is in **`%r2`** per the
/// s390x ELF ABI (same as the Linux kernel / KVM guest entry convention). A plain Rust `fn`
/// sibling could use a different convention on some targets; do not rely on it.
#[unsafe(no_mangle)]
extern "C" fn s390x_halt_after_init_dispatch_in_r2(dispatch: u64) -> ! {
    unsafe {
        core::arch::asm!(
            "diag %r4, %r5, {fc}",
            in("r4") VmAction::Halt as u64,
            in("r5") 0u64,
            in("r2") dispatch,
            fc = const S390X_HYPERLIGHT_DIAG_IO,
            options(nostack),
        );
        loop {
            core::hint::spin_loop();
        }
    }
}

/// Guest entry — same parameters as amd64 (`PEB`, RNG seed, guest page size, log filter).
///
/// The host reads **`GR2`** after the first halt to set `NextAction::Call` to the dispatch entry.
#[unsafe(no_mangle)]
pub extern "C" fn entrypoint(
    peb_address: u64,
    seed: u64,
    ops: u64,
    max_log_level: u64,
    live_scratch_bytes: u64,
) -> ! {
    if S390X_BRINGUP_DEBUG_PRINTS {
        // s390x bring-up: print the PEB pointer we were given (no scratch required).
        s390x_debug_print_hex_u64(b"s390x_guest entry peb=", peb_address);
    }
    let dispatch =
        crate::generic_init(peb_address, seed, ops, max_log_level, live_scratch_bytes);
    s390x_halt_after_init_dispatch_in_r2(dispatch);
}
