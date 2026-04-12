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

// IBM Z guest runtime (stubs). Next steps: lowcore/prefix, PSW, stack pivot,
// and a dispatch path that matches the host KVM exit model (not x86 OUT/HLT).

pub mod dispatch {
    /// Host invokes this for each guest function call. Real s390x code will
    /// need stack/PSW discipline and a VM-exit back to the host (like amd64’s
    /// asm wrapper around `internal_dispatch_function`).
    #[unsafe(no_mangle)]
    pub extern "C" fn dispatch_function() {
        unimplemented!("s390x dispatch_function")
    }
}

/// Guest entry — same parameters as amd64 (`PEB`, RNG seed, guest page size, log filter).
#[unsafe(no_mangle)]
pub extern "C" fn entrypoint(
    _peb_address: u64,
    _seed: u64,
    _ops: u64,
    _max_log_level: u64,
) -> ! {
    unimplemented!("s390x entrypoint")
}
