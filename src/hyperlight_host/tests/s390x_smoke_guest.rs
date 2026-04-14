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

//! Loads the minimal `s390x_smoke` Rust guest (`src/tests/rust_guests/s390x_smoke`) on IBM z only.
//!
//! Build the ELF first: `just build-and-move-s390x-smoke-guest`, then run:
//! `cargo test -p hyperlight-host s390x_smoke_guest -- --ignored`

#[cfg(target_arch = "s390x")]
mod s390x_smoke {
    use std::path::Path;

    use hyperlight_host::{GuestBinary, UninitializedSandbox};
    use hyperlight_testing::s390x_smoke_guest_as_string;
    use serial_test::serial;

    fn smoke_elf_path() -> String {
        let path = s390x_smoke_guest_as_string().expect("s390x_smoke path");
        assert!(
            Path::new(&path).is_file(),
            "missing ELF at {path}; run `just build-and-move-s390x-smoke-guest` (profile must match this test: debug vs release)"
        );
        path
    }

    #[test]
    #[ignore = "needs smoke ELF + KVM; run with: cargo test -p hyperlight-host s390x_smoke_guest -- --ignored"]
    #[serial]
    fn uninitialized_sandbox_new_loads_smoke_guest() {
        let path = smoke_elf_path();
        UninitializedSandbox::new(GuestBinary::FilePath(path), None)
            .expect("UninitializedSandbox::new");
    }

    #[test]
    #[ignore = "needs smoke ELF + KVM; run with: cargo test -p hyperlight-host s390x_smoke_guest -- --ignored"]
    #[serial]
    fn evolve_smoke_guest_to_multiuse_sandbox() {
        let path = smoke_elf_path();
        let u = UninitializedSandbox::new(GuestBinary::FilePath(path), None)
            .expect("UninitializedSandbox::new");
        let _multi = u.evolve().expect("evolve");
    }
}

#[cfg(not(target_arch = "s390x"))]
#[test]
fn s390x_smoke_guest_tests_are_s390x_only() {
    // Keeps this integration-test crate non-empty on other targets so `cargo test -p hyperlight-host`
    // does not special-case a zero-test binary.
}
