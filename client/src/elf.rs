// References:
// https://github.com/n-eiling/cuda-fatbin-decompression and https://github.com/RWTH-ACS/cricket
// https://github.com/jiegec/fatbinary
// https://jia.je/software/2023/10/17/clang-cuda-support/#fatbin
// https://github.com/VivekPanyam/cudaparsers
// https://github.com/llvm/llvm-project/blob/main/clang/lib/Interpreter/DeviceOffload.cpp

use std::borrow::Cow;
use std::ffi::{c_int, c_uint, c_ulonglong, c_ushort, CStr};
use std::{ptr, slice};

use elf::endian::NativeEndian;
use elf::ElfBytes;
use lz4_flex::decompress as decompress_lz4;

#[repr(C, packed)]
#[derive(Debug)]
pub struct FatBinaryWrapper {
    magic: c_int,
    version: c_int,
    data: *const c_ulonglong,
    filename_or_fatbins: usize,
}

impl FatBinaryWrapper {
    pub fn unwrap(&self) -> *const FatBinaryHeader {
        match self {
            Self {
                magic: 0x466243B1,
                version: 1,
                filename_or_fatbins: 0,
                ..
            } => {}
            Self {
                magic: 0x466243B1,
                version: 2,
                filename_or_fatbins: 1..,
                ..
            } => {}
            _ => panic!("invalid fatbin wrapper: {self:#x?}"),
        }
        self.data.cast()
    }
}

#[repr(C, align(8))]
#[derive(Debug)]
pub struct FatBinaryHeader {
    magic: c_uint,
    version: c_ushort,
    header_size: c_ushort,
    fat_size: c_ulonglong,
}

const _: () = assert!(size_of::<FatBinaryHeader>() == 16);

impl FatBinaryHeader {
    fn validate(&self) {
        let Self {
            magic: 0xBA55ED50,
            version: 1,
            header_size: 16,
            ..
        } = self
        else {
            panic!("invalid fatbin header: {self:#x?}");
        };
    }

    pub fn is_fat_binary(image: *const u8) -> bool {
        let header: &Self = unsafe { &*image.cast() };
        matches!(
            header,
            Self {
                magic: 0xBA55ED50,
                version: 1,
                header_size: 16,
                ..
            }
        )
    }

    pub fn entire_len(&self) -> usize {
        self.validate();
        self.header_size as usize + self.fat_size as usize
    }

    fn code_iter(&self) -> CodeIter {
        self.validate();
        let payload: *const u8 = ptr::from_ref(self).wrapping_add(1).cast();
        let end = payload.wrapping_add(self.fat_size as usize);
        CodeIter { payload, end }
    }

    pub fn validate_code(&self) {
        for code in self.code_iter() {
            validate_cubin(&code.data);
        }
    }

    pub fn find_kernel_params(
        &self,
        name: &str,
        target_arch: Option<u32>,
    ) -> Box<[KernelParamInfo]> {
        let mut selected: Option<(u8, u32, u32, Box<[KernelParamInfo]>)> = None;
        for cubin in self.code_iter() {
            let Some(params) = try_find_kernel_params(&cubin.data, name) else {
                continue;
            };
            let rank = match target_arch {
                Some(target) if cubin.arch <= target => (0, target - cubin.arch),
                Some(target) => (1, cubin.arch - target),
                None => (0, 0),
            };
            let replace = selected
                .as_ref()
                .map(|(kind, distance, ..)| rank < (*kind, *distance))
                .unwrap_or(true);
            if replace {
                selected = Some((rank.0, rank.1, cubin.arch, params));
            }
        }
        let Some((_, _, arch, params)) = selected else {
            panic!("Failed to find kernel params for {name}");
        };
        log::debug!("kernel {name} parameter metadata selected sm_{arch}");
        params
    }
}

struct CodeObject {
    arch: u32,
    data: Cow<'static, [u8]>,
}

struct CodeIter {
    payload: *const u8,
    end: *const u8,
}

impl Iterator for CodeIter {
    type Item = CodeObject;

    fn next(&mut self) -> Option<Self::Item> {
        let Self { payload, end } = *self;
        if payload.wrapping_add(size_of::<CodeHeader>()) >= end {
            assert_eq!(payload, end);
            return None;
        }
        let header: &CodeHeader = unsafe { &*payload.cast() };
        let (kind, compression) = header.validate();
        let code = payload.wrapping_add(header.header_size as usize);
        let code_size = header.code_size as usize;
        let code_end = code.wrapping_add(code_size);
        assert!(code_end <= end);
        self.payload = code_end;
        // TODO: We don't support parsing PTX yet. Eventually we want to return (kind, code).
        if let CodeKind::Ptx = kind {
            return self.next();
        }
        let code = unsafe { slice::from_raw_parts(code, code_size) };
        let data = if let Some(compression) = compression {
            let input = &code[..header.compressed_size as usize];
            let size = header.decompressed_size as usize;
            let decompressed = match compression {
                Compression::Lz4 => decompress_lz4(input, size).unwrap(),
                Compression::Zstd => zstd::bulk::decompress(input, size).unwrap(),
            };
            assert_eq!(decompressed.len(), size);
            Cow::Owned(decompressed)
        } else {
            Cow::Borrowed(code)
        };
        Some(CodeObject {
            arch: header.arch,
            data,
        })
    }
}

enum CodeKind {
    Ptx = 1,
    Elf = 2,
}

enum Compression {
    Lz4,
    Zstd,
}

#[repr(C, packed)]
#[derive(Debug)]
pub struct CodeHeader {
    kind: u16,
    __unknown02: u16,
    header_size: u32,
    code_size: u32,
    __unknown0c: u32,
    compressed_size: u32,
    options_offset: u32,
    minor: u16,
    major: u16,
    arch: u32,
    name_offset: u32,
    name_size: u32,
    flags: u32,
    __unknown2c: u32,
    __unknown30: u32,
    __unknown34: u32,
    decompressed_size: u32,
    __unknown3c: u32,
}

const _: () = assert!(size_of::<CodeHeader>() == 0x40);

impl CodeHeader {
    fn validate(&self) -> (CodeKind, Option<Compression>) {
        let bail = || panic!("invalid fatbin code header: {self:#x?}");
        let Self {
            __unknown02: 0x101,
            __unknown0c: 0,
            name_offset: 0,
            name_size: 0,
            __unknown2c: 0,
            __unknown30: 0,
            __unknown34: 0,
            __unknown3c: 0,
            ..
        } = self
        else {
            bail()
        };
        let kind = match self {
            Self {
                kind: 1,
                header_size: 0x48 | 0x50,
                options_offset: 0x40,
                ..
            } => CodeKind::Ptx,
            Self {
                kind: 2,
                header_size: 0x40 | 0x48 | 0x70,
                options_offset: 0 | 0x40,
                ..
            } => CodeKind::Elf,
            _ => bail(),
        };
        let compression = match self {
            Self {
                compressed_size: 0,
                flags: 0x11,
                decompressed_size: 0,
                ..
            } => None,
            Self {
                compressed_size: 1..,
                flags: 0x2011,
                decompressed_size: 1..,
                ..
            } => Some(Compression::Lz4),
            // flag 0x100000 is unknown
            Self {
                compressed_size: 1..,
                flags: 0x102011,
                decompressed_size: 1..,
                ..
            } => Some(Compression::Lz4),
            Self {
                compressed_size: 1..,
                flags: 0x8011 | 0x1008011,
                decompressed_size: 1..,
                ..
            } => Some(Compression::Zstd),
            _ => bail(),
        };
        (kind, compression)
    }
}

pub fn elf_len(cubin: *const u8) -> usize {
    let cubin = unsafe { slice::from_raw_parts(cubin, u32::MAX as _) }; // HACK
    let file = ElfBytes::<NativeEndian>::minimal_parse(cubin).unwrap();
    let shend = file.ehdr.e_shoff + (file.ehdr.e_shentsize * file.ehdr.e_shnum) as u64;
    let phend = file.ehdr.e_phoff + (file.ehdr.e_phentsize * file.ehdr.e_phnum) as u64;
    assert!(shend <= phend);
    phend as usize
}

pub fn module_image_len(image: *const u8) -> usize {
    if FatBinaryHeader::is_fat_binary(image) {
        let header: &FatBinaryHeader = unsafe { &*image.cast() };
        header.entire_len()
    } else if unsafe { slice::from_raw_parts(image, 4) } == b"\x7fELF" {
        elf_len(image)
    } else {
        unsafe { CStr::from_ptr(image.cast()) }
            .to_bytes_with_nul()
            .len()
    }
}

fn validate_cubin(cubin: &[u8]) {
    let file = ElfBytes::<NativeEndian>::minimal_parse(cubin).unwrap();
    let Ok((Some(shdrs), Some(strtab))) = file.section_headers_with_strtab() else {
        panic!("failed to parse section headers and string table")
    };
    for shdr in shdrs.iter() {
        let name = strtab
            .get(shdr.sh_name as usize)
            .expect("Failed to get section name");
        if !name.starts_with(".nv.info.") {
            continue;
        }
        let Ok((section, None)) = file.section_data(&shdr) else {
            panic!("Failed to read section or it is compressed");
        };
        let _ = parse_params(section, name);
    }
}

pub fn find_kernel_params_or_empty(cubin: &[u8], name: &str) -> Box<[KernelParamInfo]> {
    let Ok(file) = ElfBytes::<NativeEndian>::minimal_parse(cubin) else {
        log::debug!("kernel parameter metadata unavailable for {name}");
        return Box::default();
    };
    try_find_kernel_params_in_file(&file, name)
        .unwrap_or_else(|| panic!("Failed to find section header"))
}

fn try_find_kernel_params(cubin: &[u8], name: &str) -> Option<Box<[KernelParamInfo]>> {
    let Ok(file) = ElfBytes::<NativeEndian>::minimal_parse(cubin) else {
        return None;
    };
    try_find_kernel_params_in_file(&file, name)
}

fn try_find_kernel_params_in_file(
    file: &ElfBytes<'_, NativeEndian>,
    name: &str,
) -> Option<Box<[KernelParamInfo]>> {
    let Ok(Some(shdr)) = file.section_header_by_name(&[".nv.info.", name].concat()) else {
        return None;
    };
    let Ok((section, None)) = file.section_data(&shdr) else {
        panic!("Failed to read section or it is compressed");
    };
    Some(parse_params(section, name))
}

fn parse_params(nvinfo: &[u8], name: &str) -> Box<[KernelParamInfo]> {
    let mut i = 0;
    let (mut param_bytes, mut is_cbank) = (0, false);
    let mut params = Vec::new();
    while i < nvinfo.len() {
        debug_assert!(i % 4 == 0);
        let [format, attr, b0, b1] = nvinfo[i..i + 4] else {
            unreachable!()
        };
        i += 4;
        match format {
            EIFMT_NVAL => {
                assert_eq!(b0, 0);
                assert_eq!(b1, 0);
            }
            EIFMT_BVAL => {
                assert_eq!(b1, 0);
            }
            EIFMT_HVAL => {
                let val = u16::from_le_bytes([b0, b1]);
                match attr {
                    EIATTR_SMEM_PARAM_SIZE => (param_bytes, is_cbank) = (val, false),
                    EIATTR_CBANK_PARAM_SIZE => (param_bytes, is_cbank) = (val, true),
                    _ => {}
                }
            }
            EIFMT_SVAL => {
                let len = u16::from_le_bytes([b0, b1]) as usize;
                debug_assert!(len.is_multiple_of(4));
                if attr == EIATTR_KPARAM_INFO {
                    assert_eq!(len, size_of::<KernelParamInfo>());
                    assert!(i + len <= nvinfo.len());
                    let param = nvinfo.as_ptr().wrapping_add(i).cast::<KernelParamInfo>();
                    params.push(unsafe { *param });
                }
                i += len;
            }
            _ => panic!("Unknown format {format:#04x} at offset {i:#x}"),
        }
    }

    let mut params = params.into_boxed_slice();
    params.sort_unstable_by_key(|param| param.ordinal);
    for (i, param) in params.iter().enumerate() {
        if !param.has_expected_encoding() {
            log::debug!(
                "kernel {name} uses extended parameter metadata: ordinal={} offset={} size={} raw={:#x}",
                param.ordinal,
                param.offset,
                param.size(),
                param.__rest
            );
        }
        assert_eq!(i, param.ordinal as usize, "{params:#x?}");
        assert_eq!(is_cbank, param.is_cbank());
        match params.get(i + 1) {
            Some(next) => assert!(param.offset + param.size() <= next.offset),
            None => assert_eq!(param.offset + param.size(), param_bytes),
        }
    }
    params
}

const EIFMT_NVAL: u8 = 0x01;
const EIFMT_BVAL: u8 = 0x02;
const EIFMT_HVAL: u8 = 0x03;
const EIFMT_SVAL: u8 = 0x04;

const EIATTR_KPARAM_INFO: u8 = 0x17;
const EIATTR_SMEM_PARAM_SIZE: u8 = 0x18;
const EIATTR_CBANK_PARAM_SIZE: u8 = 0x19;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct KernelParamInfo {
    index: u32,
    ordinal: u16,
    pub offset: u16,
    __rest: u32,
}

const _: () = assert!(size_of::<KernelParamInfo>() == 0x0c);

#[cfg(target_endian = "little")]
impl KernelParamInfo {
    pub fn size(&self) -> u16 {
        (self.__rest >> 18) as _
    }

    fn is_cbank(&self) -> bool {
        self.__rest & (1 << 17) == 0
    }

    fn has_expected_encoding(&self) -> bool {
        let log_alignment = self.__rest & 0xff;
        let space = (self.__rest >> 8) & 0x0f;
        let cbank = (self.__rest >> 12) & 0x1f;
        self.index == 0 && log_alignment == 0 && space == 0 && cbank == 0x1f
    }
}
