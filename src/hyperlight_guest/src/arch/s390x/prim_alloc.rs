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

// amd64 uses `lock xadd` on the allocator word in guest memory. s390x would use
// an atomic read-modify-write sequence (e.g. compare-and-swap loop) per the
// z/Architecture memory model—not copied here until guest + host agree on the
// same allocator protocol.

// There are no notable architecture-specific safety considerations
// here, and the general conditions are documented in the
// architecture-independent re-export in prim_alloc.rs
#[allow(clippy::missing_safety_doc)]
pub unsafe fn alloc_phys_pages(_n: u64) -> u64 {
    unimplemented!("s390x alloc_phys_pages")
}
