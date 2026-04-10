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

// Hyperlight’s x86 guest uses CPU I/O port OUT instructions; KVM reports those
// as “I/O exit” with a port number and data. s390x has no PC-style I/O ports:
// the host must recognize a different *trap* (e.g. DIAG, SVC, or a defined
// MMIO region) and map it to the same logical `OutBAction` channel.
//
// Step 1 (this change): same stub as aarch64 so the crate builds. Step later:
// implement `out32` + host KVM exit handling as a matched pair.

/// Deliver a 32-bit value to the hypervisor on the logical “port” used for guest→host messages.
pub(crate) unsafe fn out32(_port: u16, _val: u32) {
    unimplemented!("s390x out32: wire to KVM exit path with host HyperlightVm")
}
