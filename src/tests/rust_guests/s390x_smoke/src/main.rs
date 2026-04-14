/*
Copyright 2025 The Hyperlight Authors.

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

// Minimal Hyperlight guest for s390x bring-up (no cargo-hyperlight / custom target).
#![no_std]
#![no_main]

extern crate alloc;

// Pull in `hyperlight-guest-bin` so its `#[panic_handler]`, `#[global_allocator]`, and
// `entrypoint` are linked (this crate does not reference them directly).
use hyperlight_guest_bin as _;

use alloc::vec::Vec;

use hyperlight_common::flatbuffer_wrappers::function_call::FunctionCall;
use hyperlight_common::flatbuffer_wrappers::guest_error::ErrorCode;
use hyperlight_guest::bail;
use hyperlight_guest::error::Result;

// `generic_init` in `hyperlight-guest-bin` always calls musl `srand` when the `libc`
// feature is enabled; with `default-features = false` there is no bundled C library on
// s390x (`build.rs` only builds musl for x86_64). Stub for minimal smoke guests.
#[unsafe(no_mangle)]
pub extern "C" fn srand(_seed: u32) {}

#[unsafe(no_mangle)]
pub extern "C" fn hyperlight_main() {}

#[unsafe(no_mangle)]
pub fn guest_dispatch_function(function_call: FunctionCall) -> Result<Vec<u8>> {
    let function_name = function_call.function_name;
    bail!(ErrorCode::GuestFunctionNotFound => "{function_name}");
}
