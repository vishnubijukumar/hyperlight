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

pub mod handle {
    /// Reserved for parity with amd64; no vector delivery path on s390x yet.
    pub static HANDLERS: [core::sync::atomic::AtomicU64; 31] =
        [const { core::sync::atomic::AtomicU64::new(0) }; 31];
}
