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

#[cfg(feature = "mem_profile")]
use std::sync::Arc;

#[cfg(target_arch = "aarch64")]
use goblin::elf::reloc::{R_AARCH64_NONE, R_AARCH64_RELATIVE};
#[cfg(target_arch = "x86_64")]
use goblin::elf::reloc::{R_X86_64_NONE, R_X86_64_RELATIVE};
#[cfg(target_arch = "s390x")]
use goblin::elf::sym::Sym;
use goblin::elf::{Elf, ProgramHeaders, Reloc};
#[cfg(feature = "nanvix-unstable")]
use goblin::elf32::program_header::PT_LOAD;
#[cfg(not(feature = "nanvix-unstable"))]
use goblin::elf64::program_header::PT_LOAD;

use super::exe::LoadInfo;
use crate::{Result, log_then_return, new_error};

#[cfg(feature = "mem_profile")]
struct ResolvedSectionHeader {
    name: String,
    addr: u64,
    offset: u64,
    size: u64,
}

pub(crate) struct ElfInfo {
    payload: Vec<u8>,
    phdrs: ProgramHeaders,
    #[cfg(feature = "mem_profile")]
    shdrs: Vec<ResolvedSectionHeader>,
    entry: u64,
    relocs: Vec<Reloc>,
    /// The hyperlight version string embedded by `hyperlight-guest-bin`, if
    /// present. Used to detect version/ABI mismatches between guest and host.
    guest_bin_version: Option<String>,
    /// Dynamic symbols for `R_390_64` (and similar) relocations when loading guests on s390x.
    #[cfg(target_arch = "s390x")]
    s390x_dynsyms: Vec<Sym>,
}

#[cfg(feature = "mem_profile")]
struct UnwindInfo {
    payload: Vec<u8>,
    load_addr: u64,
    va_size: u64,
    base_svma: u64,
    shdrs: Vec<ResolvedSectionHeader>,
}

#[cfg(feature = "mem_profile")]
impl super::exe::UnwindInfo for UnwindInfo {
    fn as_module(&self) -> framehop::Module<Vec<u8>> {
        framehop::Module::new(
            // TODO: plumb through a name from from_file if this
            // came from a file
            "guest".to_string(),
            self.load_addr..self.load_addr + self.va_size,
            self.load_addr,
            self,
        )
    }
    fn hash(&self) -> blake3::Hash {
        blake3::hash(&self.payload)
    }
}

#[cfg(feature = "mem_profile")]
impl UnwindInfo {
    fn resolved_section_header(&self, name: &[u8]) -> Option<&ResolvedSectionHeader> {
        self.shdrs
            .iter()
            .find(|&sh| sh.name.as_bytes()[0..core::cmp::min(name.len(), sh.name.len())] == *name)
    }
}

#[cfg(feature = "mem_profile")]
impl framehop::ModuleSectionInfo<Vec<u8>> for &UnwindInfo {
    fn base_svma(&self) -> u64 {
        self.base_svma
    }
    fn section_svma_range(&mut self, name: &[u8]) -> Option<std::ops::Range<u64>> {
        let shdr = self.resolved_section_header(name)?;
        Some(shdr.addr..shdr.addr + shdr.size)
    }
    fn section_data(&mut self, name: &[u8]) -> Option<Vec<u8>> {
        if name == b".eh_frame" && self.resolved_section_header(b".debug_frame").is_some() {
            /* Rustc does not always emit enough information for stack
             * unwinding in .eh_frame, presumably because we use panic =
             * abort in the guest. Framehop defaults to ignoring
             * .debug_frame if .eh_frame exists, but we want the opposite
             * behaviour here, since .debug_frame will actually contain
             * frame information whereas .eh_frame often doesn't because
             * of the aforementioned behaviour.  Consequently, we hack
             * around this by pretending that .eh_frame doesn't exist if
             * .debug_frame does. */
            return None;
        }
        let shdr = self.resolved_section_header(name)?;
        Some(self.payload[shdr.offset as usize..(shdr.offset + shdr.size) as usize].to_vec())
    }
}

impl ElfInfo {
    pub(crate) fn new(bytes: &[u8]) -> Result<Self> {
        let elf = Elf::parse(bytes)?;
        let relocs = elf.dynrels.iter().chain(elf.dynrelas.iter()).collect();
        if !elf
            .program_headers
            .iter()
            .any(|phdr| phdr.p_type == PT_LOAD)
        {
            log_then_return!("ELF must have at least one PT_LOAD header");
        }

        // Look for the hyperlight version note embedded by
        // hyperlight-guest-bin.
        let guest_bin_version = Self::read_version_note(&elf, bytes);

        #[cfg(target_arch = "s390x")]
        let s390x_dynsyms: Vec<Sym> = elf.dynsyms.iter().collect();

        Ok(ElfInfo {
            payload: bytes.to_vec(),
            phdrs: elf.program_headers,
            #[cfg(feature = "mem_profile")]
            shdrs: elf
                .section_headers
                .iter()
                .filter_map(|sh| {
                    Some(ResolvedSectionHeader {
                        name: elf.shdr_strtab.get_at(sh.sh_name)?.to_string(),
                        addr: sh.sh_addr,
                        offset: sh.sh_offset,
                        size: sh.sh_size,
                    })
                })
                .collect(),
            entry: elf.entry,
            relocs,
            guest_bin_version,
            #[cfg(target_arch = "s390x")]
            s390x_dynsyms,
        })
    }

    /// Read the hyperlight version note from the ELF binary
    fn read_version_note<'a>(elf: &Elf<'a>, bytes: &'a [u8]) -> Option<String> {
        use hyperlight_common::version_note::{
            HYPERLIGHT_NOTE_NAME, HYPERLIGHT_NOTE_TYPE, HYPERLIGHT_VERSION_SECTION,
        };

        let notes = elf.iter_note_sections(bytes, Some(HYPERLIGHT_VERSION_SECTION))?;
        for note in notes {
            let Ok(note) = note else { continue };
            if note.name == HYPERLIGHT_NOTE_NAME && note.n_type == HYPERLIGHT_NOTE_TYPE {
                let desc = core::str::from_utf8(note.desc).ok()?;
                return Some(desc.trim_end_matches('\0').to_string());
            }
        }
        None
    }

    pub(crate) fn entrypoint_va(&self) -> u64 {
        self.entry
    }

    /// Returns the hyperlight version string embedded in the guest binary, if
    /// present. Used to detect version/ABI mismatches between guest and host.
    pub(crate) fn guest_bin_version(&self) -> Option<&str> {
        self.guest_bin_version.as_deref()
    }

    /// SVMA (ELF virtual address) of the `dispatch_function` symbol in the static symbol table,
    /// when present. Used on Linux s390x to derive the guest dispatch entry GPA independently of
    /// GR2 after the init halt (KVM has been observed to report a bogus value there while the
    /// loaded image and GOT are correct).
    pub(crate) fn dispatch_function_svma(&self) -> Option<u64> {
        let elf = Elf::parse(&self.payload).ok()?;
        elf.syms.iter().filter(|s| s.st_name != 0).find_map(|s| {
            (elf.strtab.get_at(s.st_name) == Some("dispatch_function")).then_some(s.st_value)
        })
    }

    /// SVMA of `_GLOBAL_OFFSET_TABLE_` when present in the static or dynamic symbol table.
    ///
    /// On Linux s390x the compiler keeps the GOT pointer in **GPR12** across calls; the host must
    /// restore it when entering `dispatch_function` if KVM does not preserve it across the init
    /// halt exit (same class of issue as bogus GR2 after `KVM_EXIT_S390_SIEIC`).
    pub(crate) fn global_offset_table_svma(&self) -> Option<u64> {
        let elf = Elf::parse(&self.payload).ok()?;
        for s in elf.syms.iter().filter(|s| s.st_name != 0) {
            if elf.strtab.get_at(s.st_name) == Some("_GLOBAL_OFFSET_TABLE_") {
                return Some(s.st_value);
            }
        }
        for s in elf.dynsyms.iter().filter(|s| s.st_name != 0) {
            if elf.dynstrtab.get_at(s.st_name) == Some("_GLOBAL_OFFSET_TABLE_") {
                return Some(s.st_value);
            }
        }
        None
    }

    pub(crate) fn get_base_va(&self) -> u64 {
        #[allow(clippy::unwrap_used)] // guaranteed not to panic because of the check in new()
        let min_phdr = self
            .phdrs
            .iter()
            .find(|phdr| phdr.p_type == PT_LOAD)
            .unwrap();
        min_phdr.p_vaddr
    }
    pub(crate) fn get_va_size(&self) -> usize {
        #[allow(clippy::unwrap_used)] // guaranteed not to panic because of the check in new()
        let max_phdr = self
            .phdrs
            .iter()
            .rev()
            .find(|phdr| phdr.p_type == PT_LOAD)
            .unwrap();
        (max_phdr.p_vaddr + max_phdr.p_memsz - self.get_base_va()) as usize
    }
    pub(crate) fn load_at(self, load_addr: usize, target: &mut [u8]) -> Result<LoadInfo> {
        let base_va = self.get_base_va();
        for phdr in self.phdrs.iter().filter(|phdr| phdr.p_type == PT_LOAD) {
            let start_va = (phdr.p_vaddr - base_va) as usize;
            let payload_offset = phdr.p_offset as usize;
            let payload_len = phdr.p_filesz as usize;
            target[start_va..start_va + payload_len]
                .copy_from_slice(&self.payload[payload_offset..payload_offset + payload_len]);
            target[start_va + payload_len..start_va + phdr.p_memsz as usize].fill(0);
        }
        let get_addend = |name: &str, r: &Reloc| {
            r.r_addend
                .ok_or_else(|| new_error!("{} missing addend", name))
        };
        for r in self.relocs.iter() {
            #[cfg(target_arch = "aarch64")]
            match r.r_type {
                R_AARCH64_RELATIVE => {
                    let addend = get_addend("R_AARCH64_RELATIVE", r)?;
                    target[r.r_offset as usize..r.r_offset as usize + 8]
                        .copy_from_slice(&(load_addr as i64 + addend).to_le_bytes());
                }
                R_AARCH64_NONE => {}
                _ => {
                    log_then_return!("unsupported aarch64 relocation {}", r.r_type);
                }
            }
            #[cfg(target_arch = "x86_64")]
            match r.r_type {
                R_X86_64_RELATIVE => {
                    let addend = get_addend("R_X86_64_RELATIVE", r)?;
                    target[r.r_offset as usize..r.r_offset as usize + 8]
                        .copy_from_slice(&(load_addr as i64 + addend).to_le_bytes());
                }
                R_X86_64_NONE => {}
                _ => {
                    log_then_return!("unsupported x86_64 relocation {}", r.r_type);
                }
            }
            #[cfg(target_arch = "s390x")]
            {
                // Values from System z ELF ABI (binutils `include/elf/s390.h`). s390x uses
                // big-endian pointer words in the loaded image.
                const R_390_NONE: u32 = 0;
                const R_390_RELATIVE: u32 = 12;
                /// Direct 64-bit relocation: `S + A` (symbol runtime address + addend).
                const R_390_64: u32 = 22;
                match r.r_type {
                    R_390_RELATIVE => {
                        let addend = get_addend("R_390_RELATIVE", r)?;
                        let v = (load_addr as i64 + addend) as u64;
                        target[r.r_offset as usize..r.r_offset as usize + 8]
                            .copy_from_slice(&v.to_be_bytes());
                    }
                    R_390_64 => {
                        let addend = get_addend("R_390_64", r)?;
                        let sym = self.s390x_dynsyms.get(r.r_sym).ok_or_else(|| {
                            new_error!("R_390_64: symbol index {} out of range", r.r_sym)
                        })?;
                        if sym.is_import() {
                            log_then_return!(
                                "R_390_64: unresolved dynamic import (symbol index {})",
                                r.r_sym
                            );
                        }
                        let sym_addr = load_addr as i64 + sym.st_value as i64 - base_va as i64;
                        let v = (sym_addr + addend) as u64;
                        target[r.r_offset as usize..r.r_offset as usize + 8]
                            .copy_from_slice(&v.to_be_bytes());
                    }
                    R_390_NONE => {}
                    _ => {
                        log_then_return!("unsupported s390x relocation {}", r.r_type);
                    }
                }
            }
        }
        cfg_if::cfg_if! {
            if #[cfg(feature = "mem_profile")] {
                let va_size = self.get_va_size() as u64;
                let base_svma = self.get_base_va();
                Ok(LoadInfo {
                    info: Arc::new(UnwindInfo {
                        payload: self.payload,
                        load_addr: load_addr as u64,
                        va_size,
                        base_svma,
                        shdrs: self.shdrs,
                    })
                })
            } else {
                Ok(LoadInfo {})
            }
        }
    }
}

#[cfg(all(test, target_arch = "s390x"))]
mod s390x_reloc_tests {
    use goblin::elf::Elf;

    use super::ElfInfo;
    use crate::mem::layout::SandboxMemoryLayout;

    /// `generic_init` returns the address loaded from the GOT (`lgrl %r2,…`); it must match
    /// `load_addr + addend` for every `R_390_RELATIVE` slot that targets `dispatch_function`.
    #[test]
    fn simpleguest_dispatch_got_slots_use_load_base() {
        const R_390_RELATIVE: u32 = 12;

        let path =
            hyperlight_testing::simple_guest_as_string().expect("locate simpleguest binary");
        let bytes = std::fs::read(&path).expect("read simpleguest");
        let elf = Elf::parse(&bytes).expect("parse ELF");

        let dispatch_svma = elf
            .syms
            .iter()
            .filter(|s| s.st_name != 0)
            .find_map(|s| {
                (elf.strtab.get_at(s.st_name) == Some("dispatch_function")).then_some(s.st_value)
            })
            .expect("dispatch_function symbol in symtab");

        let info = ElfInfo::new(&bytes).expect("ElfInfo::new");
        let load_addr = SandboxMemoryLayout::BASE_ADDRESS;
        let mut mem = vec![0u8; info.get_va_size()];
        info.load_at(load_addr, &mut mem).expect("load_at");

        let mut matched = 0usize;
        for r in elf.dynrelas.iter() {
            if r.r_type != R_390_RELATIVE {
                continue;
            }
            let Some(addend) = r.r_addend else {
                continue;
            };
            if addend != dispatch_svma as i64 {
                continue;
            }
            let o = r.r_offset as usize;
            let word = u64::from_be_bytes(mem[o..o + 8].try_into().expect("GOT word in range"));
            assert_eq!(
                word,
                load_addr as u64 + addend as u64,
                "R_390_RELATIVE at {o:#x}: GOT must hold load_addr + addend (BE)"
            );
            matched += 1;
        }
        assert!(
            matched > 0,
            "expected at least one R_390_RELATIVE GOT slot for dispatch_function (addend={dispatch_svma:#x})"
        );
    }

    #[test]
    fn simpleguest_global_offset_table_sym_matches_symtab() {
        let path =
            hyperlight_testing::simple_guest_as_string().expect("locate simpleguest binary");
        let bytes = std::fs::read(&path).expect("read simpleguest");
        let elf = Elf::parse(&bytes).expect("parse ELF");
        let got_svma = elf
            .syms
            .iter()
            .filter(|s| s.st_name != 0)
            .find_map(|s| {
                (elf.strtab.get_at(s.st_name) == Some("_GLOBAL_OFFSET_TABLE_")).then_some(s.st_value)
            })
            .expect("_GLOBAL_OFFSET_TABLE_ in symtab");

        let info = ElfInfo::new(&bytes).expect("ElfInfo::new");
        assert_eq!(
            info.global_offset_table_svma(),
            Some(got_svma),
            "ElfInfo must resolve GOT the same as goblin symtab"
        );
    }
}
