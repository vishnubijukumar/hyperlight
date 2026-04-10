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

// s390x does not use Intel/ARM-style multilevel page tables. IBM z uses
// Dynamic Address Translation (DAT): an address-space-control element (ASCE)
// indexes region/segment/page tables with z-specific entry layouts and
// protection rules (see z/Architecture Principles of Operation, “Translation”).
//
// Hyperlight’s `TableOps` / `map` / `virt_to_phys` API is shaped around the
// x86 bring-up path. For s390x we expose the same types and stubs as aarch64
// so the crate graph compiles; a real implementation must model DAT tables
// and KVM’s view of guest memory, not reuse x86 PTE encodings.

use crate::vmem::{Mapping, TableOps, TableReadOps, Void};

pub const PAGE_SIZE: usize = 4096;
pub const PAGE_TABLE_SIZE: usize = 4096;
pub type PageTableEntry = u64;
pub type VirtAddr = u64;
pub type PhysAddr = u64;

/// # Safety
/// See `TableOps` documentation.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn map<Op: TableOps>(_op: &Op, _mapping: Mapping) {
    unimplemented!("map")
}

/// # Safety
/// See `TableReadOps` documentation.
#[allow(clippy::missing_safety_doc)]
pub unsafe fn virt_to_phys<'a, Op: TableReadOps + 'a>(
    _op: impl core::convert::AsRef<Op> + Copy + 'a,
    _address: u64,
    _len: u64,
) -> impl Iterator<Item = Mapping> + 'a {
    unimplemented!("virt_to_phys");
    #[allow(unreachable_code)]
    core::iter::empty()
}

pub trait TableMovability<Op: TableReadOps + ?Sized, TableMoveInfo> {}
impl<Op: TableOps<TableMovability = crate::vmem::MayMoveTable>> TableMovability<Op, Op::TableAddr>
    for crate::vmem::MayMoveTable
{
}
impl<Op: TableReadOps> TableMovability<Op, Void> for crate::vmem::MayNotMoveTable {}
