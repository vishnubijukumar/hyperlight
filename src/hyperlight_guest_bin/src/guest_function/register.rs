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

#[cfg(not(target_arch = "s390x"))]
use alloc::collections::BTreeMap;
use alloc::string::String;
#[cfg(target_arch = "s390x")]
use alloc::vec::Vec;

use hyperlight_common::func::{ParameterTuple, SupportedReturnType};

use super::definition::{GuestFunc, GuestFunctionDefinition};
use crate::REGISTERED_GUEST_FUNCTIONS;
use crate::guest_function::definition::AsGuestFunctionDefinition;

/// Represents the functions that the guest exposes to the host.
#[derive(Debug, Clone)]
pub struct GuestFunctionRegister<F: Copy> {
    /// Currently registered guest functions
    #[cfg(not(target_arch = "s390x"))]
    guest_functions: BTreeMap<String, GuestFunctionDefinition<F>>,
    /// Linux s390x KVM bring-up: `buddy_system_allocator::LockedHeap` uses a non-reentrant
    /// `spin::Mutex`. `BTreeMap::insert` can perform multiple internal allocations per insert;
    /// combined with the global logger / `String` growth paths this has been observed to wedge
    /// the guest during `GUEST_FUNCTION_INIT`. A flat `Vec` keeps registration to simple pushes.
    #[cfg(target_arch = "s390x")]
    guest_functions: Vec<(String, GuestFunctionDefinition<F>)>,
}

impl<F: Copy> Default for GuestFunctionRegister<F> {
    fn default() -> Self {
        Self {
            #[cfg(not(target_arch = "s390x"))]
            guest_functions: BTreeMap::new(),
            #[cfg(target_arch = "s390x")]
            guest_functions: Vec::new(),
        }
    }
}

impl<F: Copy> GuestFunctionRegister<F> {
    /// Create a new `GuestFunctionRegister`.
    pub const fn new() -> Self {
        Self {
            #[cfg(not(target_arch = "s390x"))]
            guest_functions: BTreeMap::new(),
            #[cfg(target_arch = "s390x")]
            guest_functions: Vec::new(),
        }
    }

    /// Register a new `GuestFunctionDefinition` into self.
    /// If a function with the same name already exists, it will be replaced.
    /// None is returned if the function name was not previously registered,
    /// otherwise the previous `GuestFunctionDefinition` is returned.
    pub fn register(
        &mut self,
        guest_function: GuestFunctionDefinition<F>,
    ) -> Option<GuestFunctionDefinition<F>> {
        #[cfg(not(target_arch = "s390x"))]
        {
            self.guest_functions
                .insert(guest_function.function_name.clone(), guest_function)
        }
        #[cfg(target_arch = "s390x")]
        {
            let name = guest_function.function_name.clone();
            for (n, def) in self.guest_functions.iter_mut() {
                if *n == name {
                    return Some(core::mem::replace(def, guest_function));
                }
            }
            self.guest_functions.push((name, guest_function));
            None
        }
    }

    /// Gets a `GuestFunctionDefinition` by its `name` field.
    pub fn get(&self, function_name: &str) -> Option<&GuestFunctionDefinition<F>> {
        #[cfg(not(target_arch = "s390x"))]
        {
            self.guest_functions.get(function_name)
        }
        #[cfg(target_arch = "s390x")]
        {
            self.guest_functions
                .iter()
                .find(|(n, _)| n == function_name)
                .map(|(_, g)| g)
        }
    }
}

impl GuestFunctionRegister<GuestFunc> {
    pub fn register_fn<Output, Args>(
        &mut self,
        name: impl Into<String>,
        f: impl AsGuestFunctionDefinition<Output, Args>,
    ) where
        Args: ParameterTuple,
        Output: SupportedReturnType,
    {
        let gfd = f.as_guest_function_definition(name);
        self.register(gfd);
    }
}

pub fn register_function(function_definition: GuestFunctionDefinition<GuestFunc>) {
    unsafe {
        // This is currently safe, because we are single threaded, but we
        // should find a better way to do this, see issue #808
        #[allow(static_mut_refs)]
        let gfd = &mut REGISTERED_GUEST_FUNCTIONS;
        gfd.register(function_definition);
    }
}

pub fn register_fn<Output, Args>(
    name: impl Into<String>,
    f: impl AsGuestFunctionDefinition<Output, Args>,
) where
    Args: ParameterTuple,
    Output: SupportedReturnType,
{
    unsafe {
        // This is currently safe, because we are single threaded, but we
        // should find a better way to do this, see issue #808
        #[allow(static_mut_refs)]
        let gfd = &mut REGISTERED_GUEST_FUNCTIONS;
        gfd.register_fn(name, f);
    }
}
