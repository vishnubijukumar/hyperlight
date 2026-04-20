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

#![cfg_attr(not(any(test, debug_assertions)), warn(clippy::panic))]
#![cfg_attr(not(any(test, debug_assertions)), warn(clippy::expect_used))]
#![cfg_attr(not(any(test, debug_assertions)), warn(clippy::unwrap_used))]
// clippy.toml disallows assert!/assert_eq!/assert_ne! via disallowed-macros.
// That lint is active by default, so we suppress it globally here, then
// selectively re-enable it for release host builds (feature = "std").
// Guest targets (no std) are allowed asserts — panics are contained in the
// micro-VM and cannot crash the host. Tests and debug builds are also allowed.
#![allow(clippy::disallowed_macros)]
#![cfg_attr(
    not(any(test, debug_assertions, not(feature = "std"))),
    warn(clippy::disallowed_macros)
)]
// We use Arbitrary during fuzzing, which requires std
#![cfg_attr(not(feature = "fuzzing"), no_std)]

extern crate alloc;

pub mod flatbuffer_wrappers;
/// cbindgen:ignore
/// FlatBuffers-related utilities and (mostly) generated code
#[allow(clippy::all, warnings)]
mod flatbuffers;
// cbindgen:ignore
pub mod layout;

// cbindgen:ignore
pub mod log_level;

/// cbindgen:ignore
pub mod mem;

/// cbindgen:ignore
pub mod outb;

/// cbindgen:ignore
pub mod resource;

/// cbindgen:ignore
pub mod func;

// cbindgen:ignore
pub mod vmem;

/// ELF note types for embedding hyperlight version metadata in guest binaries.
pub mod version_note;
