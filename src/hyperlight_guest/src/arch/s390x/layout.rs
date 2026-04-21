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

// Coordinate these with:
// - `hyperlight_common::layout` (MAX_GVA / MAX_GPA / scratch math)
// - `hyperlight_guest_bin::arch::s390x::layout` (once that exists)
//
// Stack bounds: same high virtual addresses as amd64/aarch64 placeholders so
// host and guest agree on a 64-bit upper region until we define an s390x map.

use core::sync::atomic::{AtomicU64, Ordering};

/// Top of the guest main stack (virtual address).
pub const MAIN_STACK_TOP_GVA: u64 = 0xffff_ff00_0000_0000;
/// Lower limit of that stack’s growth region.
pub const MAIN_STACK_LIMIT_GVA: u64 = 0xffff_fe00_0000_0000;

/// Live scratch span in bytes (matches `HostSharedMemory::mem_size` / KVM slot).
///
/// Linux KVM s390x guests do not load the snapshot page tables: `scratch_size_gva`
/// cannot use the amd64-style `MAX_GVA - offset` cell, which is not backed by the
/// scratch memslot. The host passes the rounded size in **GR6** at guest entry
/// (`HyperlightVm::initialise`); `hyperlight-guest-bin` records it here before any
/// code reads scratch metadata or `scratch_base_gpa()`.
static LIVE_SCRATCH_BYTES: AtomicU64 = AtomicU64::new(0);

/// Called from `hyperlight-guest-bin` `generic_init` before other guest setup.
#[inline]
pub fn init_live_scratch_from_host(bytes: u64) {
    LIVE_SCRATCH_BYTES.store(bytes, Ordering::Relaxed);
}

/// Size of the scratch region (bytes), as agreed with the host for this sandbox.
#[inline]
pub fn scratch_size() -> u64 {
    LIVE_SCRATCH_BYTES.load(Ordering::Relaxed)
}

pub fn scratch_base_gpa() -> u64 {
    let sz = scratch_size();
    if sz == 0 {
        return 0;
    }
    hyperlight_common::layout::scratch_base_gpa(sz as usize)
}

/// Linux KVM s390x bring-up uses identity guest addresses for scratch (GVA == GPA).
#[inline]
pub fn scratch_base_gva() -> u64 {
    scratch_base_gpa()
}
