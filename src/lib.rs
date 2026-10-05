//! Zero-copy import of externally produced GPU frames into [`wgpu`] textures.
//!
//! Browser engines, media decoders, capture pipelines, and compositors all hand
//! their output over as a platform handle to memory the GPU already holds:
//! a DMA-BUF file descriptor on Linux, an `AHardwareBuffer` on Android, an
//! `IOSurface` on Apple platforms, a shared `ID3D12Resource` handle on
//! Windows. This crate turns each of those into a [`wgpu::Texture`] on a
//! device the caller already owns, without ever routing a pixel through the
//! CPU.
//!
//! Each platform gets its own module, because the handles genuinely have
//! nothing in common: one is a file descriptor with a DRM format modifier and
//! an explicit fence, one is a reference-counted NDK object, one is a
//! `CoreFoundation` object, one is a `Win32` `HANDLE`. There is no unifying
//! trait, and inventing one would only hide the differences a caller has to
//! know about anyway.
//!
//! - [`dma_buf`] — Linux DMA-BUF, imported through Vulkan external memory
//!   (`VK_EXT_external_memory_dma_buf`) or EGL
//!   (`EGL_LINUX_DMA_BUF_EXT` plus `glEGLImageTargetTexture2DOES`), whichever
//!   backend the `wgpu` device runs on.
//! - [`ahardware_buffer`] — Android `AHardwareBuffer`, imported through Vulkan
//!   (`VK_ANDROID_external_memory_android_hardware_buffer`) and adopted by
//!   `wgpu` without a copy.
//! - [`io_surface`] — macOS and iOS `IOSurface`, packed RGBA/BGRA or
//!   biplanar 4:2:0 YCbCr one plane per texture, imported through
//!   `MTLDevice::newTextureWithDescriptor:iosurface:plane:`.
//! - [`shared_handle`] — Windows shared texture handles, imported through
//!   `ID3D12Device::OpenSharedHandle`.
//!
//! Every import reaches under `wgpu` to its `hal` layer, so the caller must
//! run on the backend that platform's import requires; each entry point says
//! which, and asserts it rather than guessing.

#[cfg(target_os = "android")]
pub mod ahardware_buffer;
#[cfg(target_os = "linux")]
pub mod dma_buf;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod io_surface;
#[cfg(target_os = "windows")]
pub mod shared_handle;
#[cfg(any(target_os = "linux", target_os = "android"))]
mod vulkan_memory;
mod ycbcr;

pub use ycbcr::{YcbcrEncoding, YcbcrMatrix, YcbcrRange};
