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

// Address bounds for the s390x (IBM z/Architecture) port.
//
// z/Architecture is a 64-bit ISA: guest virtual and real addresses are
// 64-bit. Linux on s390 uses 4 KiB base pages and Dynamic Address Translation
// (DAT), not x86-style multilevel PTEs. These constants match the *same API*
// as amd64/aarch64 in this crate so host code can compute scratch placement
// (`scratch_base_gpa` / `scratch_base_gva`) without `#[cfg]` everywhere.
//
// MAX_GVA: same “top minus one page” convention as amd64 (see amd64/layout.rs):
// avoids edge cases when code uses end pointers or inclusive bounds.
pub const MAX_GVA: usize = 0xffff_ffff_ffff_efff;

// Placeholder range for “where a snapshot page table might live” in guest VA,
// mirroring aarch64 until a dedicated s390x guest memory map exists.
pub const SNAPSHOT_PT_GVA_MIN: usize = 0xffff_8000_0000_0000;
pub const SNAPSHOT_PT_GVA_MAX: usize = 0xffff_80ff_ffff_ffff;

// MAX_GPA: upper bound used for `scratch_base_gpa` = `MAX_GPA - scratch_size + 1`.
//
// Linux s390 KVM rejects `KVM_SET_USER_MEMORY_REGION` when the exclusive end of a
// slot exceeds `kvm->arch.mem_limit` (`sclp.hamax + 1` when the VM is created).
// Scratch is mapped at the top of this range, so its end is `MAX_GPA + 1`. The
// 40-bit-style value shared with aarch64 is often **above** `mem_limit` on
// smaller LPARs or nested guests, causing `EINVAL` on the scratch slot only.
//
// Keep this within a modest span; raise when we can query `mem_limit` or
// relocate scratch for large guests.
pub const MAX_GPA: usize = 0x0000_0000_3fff_ffff; // 1 GiB - 1  →  scratch ends at 1 GiB

// Scratch sizing: amd64 counts fixed pages (TSS/IDT, PTE staging, stacks, I/O).
// s390x guests use different machinery (lowcore, prefix areas, PSW, etc.), but
// the host still needs a similarly sized scratch window until we have a
// documented s390x-specific breakdown. This matches amd64’s formula so callers
// do not hit `unimplemented!` during bring-up.
pub fn min_scratch_size(input_data_size: usize, output_data_size: usize) -> usize {
    (input_data_size + output_data_size).next_multiple_of(crate::vmem::PAGE_SIZE)
        + 12 * crate::vmem::PAGE_SIZE
}
