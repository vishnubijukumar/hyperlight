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

/// Top of the guest main stack (virtual address).
pub const MAIN_STACK_TOP_GVA: u64 = 0xffff_ff00_0000_0000;
/// Lower limit of that stack’s growth region.
pub const MAIN_STACK_LIMIT_GVA: u64 = 0xffff_fe00_0000_0000;

/// Size of the scratch region (bytes), as stored by the host in guest memory.
///
/// The host writes this at a fixed offset below `MAX_GVA` (see
/// `layout::scratch_size_gva`). We load one `u64` from that cell. Using
/// `read_volatile` avoids the compiler caching or reordering the read, which
/// matters if the host updates scratch metadata while the guest runs.
pub fn scratch_size() -> u64 {
    let addr = crate::layout::scratch_size_gva();
    unsafe { addr.cast::<u64>().read_volatile() }
}

pub fn scratch_base_gpa() -> u64 {
    hyperlight_common::layout::scratch_base_gpa(scratch_size() as usize)
}

pub fn scratch_base_gva() -> u64 {
    hyperlight_common::layout::scratch_base_gva(scratch_size() as usize)
}
