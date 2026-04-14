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

// amd64 uses `lock xadd` on the allocator word. We use `AtomicU64::fetch_add`
// so LLVM emits the appropriate z/Architecture atomic (e.g. LAAG) for the same
// bump-allocator protocol.

use core::sync::atomic::{AtomicU64, Ordering};

use hyperlight_common::flatbuffer_wrappers::guest_error::ErrorCode;

// There are no notable architecture-specific safety considerations
// here, and the general conditions are documented in the
// architecture-independent re-export in prim_alloc.rs
#[allow(clippy::missing_safety_doc)]
pub unsafe fn alloc_phys_pages(n: u64) -> u64 {
    let addr = crate::layout::allocator_gva();
    let nbytes = n * hyperlight_common::vmem::PAGE_SIZE as u64;
    let x = unsafe {
        AtomicU64::from_ptr(addr as *mut u64).fetch_add(nbytes, Ordering::SeqCst)
    };
    let max_avail = hyperlight_common::layout::MAX_GPA - hyperlight_common::vmem::PAGE_SIZE * 2;
    if x.checked_add(nbytes)
        .is_none_or(|xx| xx >= max_avail as u64)
    {
        unsafe {
            crate::exit::abort_with_code_and_message(
                &[ErrorCode::MallocFailed as u8],
                c"Out of physical memory".as_ptr(),
            )
        }
    }
    x
}
