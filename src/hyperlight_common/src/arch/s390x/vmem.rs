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

// Same x86-64 page table image in guest RAM for the sandbox protocol; the s390x CPU
// does not interpret it. Host snapshot code still needs `map` / `virt_to_phys` when
// built for `target_arch = "s390x"`.

include!("../amd64/vmem.rs");
