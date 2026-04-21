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

#[cfg_attr(target_arch = "x86_64", path = "arch/amd64/layout.rs")]
#[cfg_attr(target_arch = "x86", path = "arch/i686/layout.rs")]
#[cfg_attr(target_arch = "aarch64", path = "arch/aarch64/layout.rs")]
#[cfg_attr(target_arch = "s390x", path = "arch/s390x/layout.rs")]
mod arch;

pub use arch::{MAIN_STACK_LIMIT_GVA, MAIN_STACK_TOP_GVA};

#[cfg(target_arch = "s390x")]
static mut PREINIT_SCRATCH_META_DUMMY: u64 = 0;

/// Record the live scratch byte size from the host (s390x only). See `arch/s390x/layout.rs`.
#[cfg(target_arch = "s390x")]
pub fn init_s390_live_scratch_from_host(bytes: u64) {
    arch::init_live_scratch_from_host(bytes);
}

/// Pointer to the `u64` scratch-size cell at the top of the guest scratch region.
///
/// On amd64 this aliases a page mapped below `MAX_GVA`. On Linux s390x KVM the scratch
/// slot is a low GPA range (`scratch_base_gpa`); metadata addresses must use that base.
#[cfg(target_arch = "s390x")]
pub fn scratch_size_gva() -> *mut u64 {
    use hyperlight_common::layout::{SCRATCH_TOP_SIZE_OFFSET, scratch_base_gpa};
    let sz = arch::scratch_size();
    if sz == 0 {
        return unsafe { core::ptr::addr_of_mut!(PREINIT_SCRATCH_META_DUMMY) };
    }
    let base = scratch_base_gpa(sz as usize);
    (base + sz - SCRATCH_TOP_SIZE_OFFSET) as *mut u64
}

#[cfg(not(target_arch = "s390x"))]
pub fn scratch_size_gva() -> *mut u64 {
    use hyperlight_common::layout::{MAX_GVA, SCRATCH_TOP_SIZE_OFFSET};
    (MAX_GVA as u64 - SCRATCH_TOP_SIZE_OFFSET + 1) as *mut u64
}

#[cfg(target_arch = "s390x")]
pub fn allocator_gva() -> *mut u64 {
    use hyperlight_common::layout::{SCRATCH_TOP_ALLOCATOR_OFFSET, scratch_base_gpa};
    let sz = arch::scratch_size();
    if sz == 0 {
        return unsafe { core::ptr::addr_of_mut!(PREINIT_SCRATCH_META_DUMMY) };
    }
    let base = scratch_base_gpa(sz as usize);
    (base + sz - SCRATCH_TOP_ALLOCATOR_OFFSET) as *mut u64
}

#[cfg(not(target_arch = "s390x"))]
pub fn allocator_gva() -> *mut u64 {
    use hyperlight_common::layout::{MAX_GVA, SCRATCH_TOP_ALLOCATOR_OFFSET};
    (MAX_GVA as u64 - SCRATCH_TOP_ALLOCATOR_OFFSET + 1) as *mut u64
}

#[cfg(target_arch = "s390x")]
pub fn snapshot_pt_gpa_base_gva() -> *mut u64 {
    use hyperlight_common::layout::{SCRATCH_TOP_SNAPSHOT_PT_GPA_BASE_OFFSET, scratch_base_gpa};
    let sz = arch::scratch_size();
    if sz == 0 {
        return unsafe { core::ptr::addr_of_mut!(PREINIT_SCRATCH_META_DUMMY) };
    }
    let base = scratch_base_gpa(sz as usize);
    (base + sz - SCRATCH_TOP_SNAPSHOT_PT_GPA_BASE_OFFSET) as *mut u64
}

#[cfg(not(target_arch = "s390x"))]
pub fn snapshot_pt_gpa_base_gva() -> *mut u64 {
    use hyperlight_common::layout::{MAX_GVA, SCRATCH_TOP_SNAPSHOT_PT_GPA_BASE_OFFSET};
    (MAX_GVA as u64 - SCRATCH_TOP_SNAPSHOT_PT_GPA_BASE_OFFSET + 1) as *mut u64
}
pub use arch::{scratch_base_gpa, scratch_base_gva};

/// Returns a pointer to the guest counter u64 in scratch memory.
#[cfg(feature = "nanvix-unstable")]
pub fn guest_counter_gva() -> *const u64 {
    use hyperlight_common::layout::{MAX_GVA, SCRATCH_TOP_GUEST_COUNTER_OFFSET};
    (MAX_GVA as u64 - SCRATCH_TOP_GUEST_COUNTER_OFFSET + 1) as *const u64
}
