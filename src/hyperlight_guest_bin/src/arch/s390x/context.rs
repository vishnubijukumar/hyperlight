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

use core::mem::size_of;

/// Placeholder layout matching the amd64 guest `Context` so shared crates compile.
/// s390x does not install the x86-style exception entry path; handlers are unused on this arch.
#[repr(C)]
pub struct Context {
    pub segments: [u64; 4],
    pub fxsave: [u8; 512],
    pub gprs: [u64; 15],
    _padding: u64,
}

const _: () = assert!(size_of::<Context>() == 32 + 512 + 120 + 8);
