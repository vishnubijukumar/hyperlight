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
//!
//! `evolve()` needs KVM (`/dev/kvm` readable, typically membership in the `kvm` group).

#[cfg(target_arch = "s390x")]
mod s390x_smoke {
    use std::path::Path;

    use hyperlight_host::{GuestBinary, UninitializedSandbox, is_hypervisor_present};
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

    /// ELF parse + snapshot in `new()`, then KVM-backed `evolve()`. Single test avoids relying on
    /// `cargo test` name order (alphabetical) to run a cheap `new()`-only case before KVM.
    #[test]
    #[ignore = "needs smoke ELF + KVM; run with: cargo test -p hyperlight-host s390x_smoke_guest -- --ignored"]
    #[serial]
    fn s390x_smoke_guest_new_then_evolve() {
        let path = smoke_elf_path();
        if !is_hypervisor_present() {
            let dev_kvm = Path::new("/dev/kvm");
            panic!(
                "{}",
                if dev_kvm.exists() {
                    "KVM probe failed while /dev/kvm exists: this user likely cannot open it (device is usually root:kvm, mode 0660). \
                     Add the account to the kvm group: `sudo usermod -aG kvm \"$USER\"`, then start a new login session (or run `newgrp kvm`) and confirm `groups` includes kvm. \
                     If kvm is already listed, run with `RUST_LOG=info` and check KVM_GET_API_VERSION / KVM_CAP_USER_MEMORY logs from the hypervisor probe."
                } else {
                    "KVM required: /dev/kvm is missing (KVM not loaded or unavailable in this environment)."
                }
            );
        }
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
