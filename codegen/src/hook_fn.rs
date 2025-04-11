//! Semantic parsing of hook definitions.

use hookdef::{check_max_attributes, HookAttrs, HookFnItem, HookInjections};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use syn::{
    parse_quote, Attribute, Error, FnArg, LitInt, Meta, Pat, PatIdent, Result, ReturnType,
    Signature, Type, TypePtr,
};

use crate::utils::{
    is_async_return_type, is_const_cstr, is_shadow_desc_type, is_void_ptr, Element, ElementMode,
    PassBy,
};

pub struct HookFn {
    pub proc_id: LitInt,
    pub is_async_api: bool,
    pub is_create_shadow_desc: bool,
    pub func: Ident,
    pub result: Element,
    pub params: Box<[Element]>,
    pub sig: Signature,
    pub injections: HookInjections,
}

impl HookFn {
    pub fn parse(args: TokenStream, input: TokenStream) -> Result<Self> {
        Self::new(HookAttrs::from_macro(args)?, syn::parse2(input)?)
    }

    fn new(
        HookAttrs { proc_id, is_async_api, .. }: HookAttrs,
        HookFnItem { sig, injections }: HookFnItem,
    ) -> Result<Self> {
        if is_async_api {
            check_async_api(&sig.output, &injections)?;
        }

        let result = Element {
            name: format_ident!("result"),
            ty: match &sig.output {
                ReturnType::Default => syn::parse_quote!(()),
                ReturnType::Type(_, ty) => ty.as_ref().clone(),
            },
            mode: ElementMode::Output,
            pass_by: PassBy::InputValue,
            is_void_ptr: false,
        };

        let params = sig
            .inputs
            .iter()
            .map(|arg| parse_param(arg, is_async_api))
            .collect::<Result<Box<_>>>()?;

        if let Some(param) = params.iter().find(|param| param.mode == ElementMode::Skip) {
            if injections.server_execution.is_empty() {
                return Err(Error::new_spanned(
                    &param.name,
                    "custom execution required when skipping parameter",
                ));
            }
        }

        fn is_create_shadow_desc(func: &Ident, params: &[Element]) -> bool {
            if params.len() != 1 {
                return false;
            }
            let param = &params[0];
            if param.mode != ElementMode::Output {
                return false;
            }
            let Type::Ptr(ptr) = &param.ty else { panic!() };
            if ptr.mutability.is_some() && is_shadow_desc_type(&ptr.elem) {
                assert!(func.to_string().contains("Create"));
                true
            } else {
                false
            }
        }

        Ok(Self {
            proc_id,
            is_async_api,
            is_create_shadow_desc: is_create_shadow_desc(&sig.ident, &params),
            func: sig.ident.clone(),
            result,
            params,
            sig,
            injections,
        })
    }

    pub fn into_plain_fn(self) -> TokenStream {
        let mut sig = self.sig;

        if !self.is_async_api
            && check_async_api(&sig.output, &self.injections).is_ok()
            && self.params.iter().all(|x| x.mode == ElementMode::Input)
            && !self.func.to_string().contains("Error")
        {
            sig.ident.span().unwrap().note("this function can be `async_api`").emit();
        }

        for arg in sig.inputs.iter_mut() {
            let FnArg::Typed(arg) = arg else { panic!() };
            arg.attrs.clear();
        }
        quote! {
            #sig {
                unimplemented!()
            }
        }
    }
}

fn check_async_api(output: &ReturnType, injections: &HookInjections) -> Result<()> {
    if let Some(stmt) =
        injections.client_after_recv.first().or(injections.server_after_send.first())
    {
        return Err(Error::new_spanned(stmt, "`async_api` cannot have `after` injections"));
    }
    match output {
        ReturnType::Type(_, ty) if !is_async_return_type(ty) => {
            Err(Error::new_spanned(output, "unsupported `async_api` return type"))
        }
        _ => Ok(()),
    }
}

fn parse_param(arg: &FnArg, is_async_api: bool) -> Result<Element> {
    let FnArg::Typed(arg) = arg else { panic!() };

    // Get param name
    let Pat::Ident(PatIdent {
        by_ref: None,
        mutability: None,
        ref ident,
        subpat: None,
        ..
    }) = *arg.pat
    else {
        panic!()
    };

    let ty = arg.ty.as_ref();
    let (mode, pass_by) = if let Type::Ptr(ptr) = ty {
        check_max_attributes(&arg.attrs, 1)?;
        if let Some(attr) = arg.attrs.first() {
            parse_param_attr(attr, ptr)?
        } else if is_const_cstr(ptr) {
            (ElementMode::Input, PassBy::InputCStr)
        } else if ptr.const_token.is_some() || is_void_ptr(ptr) {
            return Err(Error::new_spanned(
                arg,
                "expected #[device] or #[host(...)]",
            ));
        } else {
            (ElementMode::Output, PassBy::SinglePtr)
        }
    } else {
        check_max_attributes(&arg.attrs, 0)?;
        (ElementMode::Input, PassBy::InputValue)
    };

    if mode == ElementMode::Output && is_async_api {
        return Err(Error::new_spanned(
            arg,
            "output parameter is not allowed for async_api",
        ));
    }

    let mut ty = ty.clone();
    let is_void_ptr = match (&pass_by, &mut ty) {
        (PassBy::InputValue, _) => false,
        (_, Type::Ptr(ref mut ptr)) if is_void_ptr(ptr) => {
            ptr.elem = Box::new(parse_quote!(u8));
            true
        }
        _ => false,
    };

    Ok(Element {
        name: ident.clone(),
        ty,
        mode,
        pass_by,
        is_void_ptr,
    })
}

fn parse_param_attr(attr: &Attribute, ptr: &TypePtr) -> Result<(ElementMode, PassBy)> {
    let location = attr.path().require_ident()?.to_string();
    if location == "host" {
        let is_void_ptr = is_void_ptr(ptr);
        let is_const_ptr = ptr.const_token.is_some();
        let mut mode = None;
        let mut len = None;
        let mut cap = None;
        if matches!(attr.meta, Meta::Path(_)) {
            // No nested meta
        } else {
            attr.parse_nested_meta(|meta| {
                match meta.path.require_ident()?.to_string().as_str() {
                    "input" => {
                        if is_const_ptr {
                            return Err(meta.error("not allowed on const pointer"));
                        }
                        mode = Some(ElementMode::Input);
                    }
                    "output" => {
                        if is_const_ptr {
                            return Err(meta.error("not allowed on const pointer"));
                        }
                        mode = Some(ElementMode::Output);
                    }
                    "len" => len = Some(meta.value()?.parse()?),
                    "cap" => cap = Some(meta.value()?.parse()?),
                    _ => return Err(meta.error("unsupported property")),
                }
                Ok(())
            })?;
        }

        let mode = if is_const_ptr {
            ElementMode::Input
        } else if let Some(mode) = mode {
            mode
        } else {
            return Err(Error::new_spanned(
                attr,
                "input/output property is required for mutable pointer",
            ));
        };

        let pass_by = if let Some(len) = len {
            PassBy::ArrayPtr { len, cap }
        } else if !is_void_ptr {
            PassBy::SinglePtr
        } else {
            return Err(Error::new_spanned(
                attr,
                "len property is required for void pointer",
            ));
        };

        Ok((mode, pass_by))
    } else if location == "device" && matches!(attr.meta, Meta::Path(_)) {
        Ok((ElementMode::Input, PassBy::InputValue))
    } else if location == "skip" && matches!(attr.meta, Meta::Path(_)) {
        Ok((ElementMode::Skip, PassBy::InputValue))
    } else {
        Err(Error::new_spanned(
            attr,
            "expected #[device] or #[host(...)]",
        ))
    }
}
