//! Apple `IOSurface` import, on macOS and iOS.
//!
//! An `IOSurface` is the system's handle to pixel memory shared between
//! processes, the CPU, the GPU, and the media engines. Metal imports one plane
//! of a surface directly with `newTextureWithDescriptor:iosurface:plane:`, and
//! `wgpu` adopts the resulting `MTLTexture` through its `hal` layer, so no
//! pixel is copied.
//!
//! Two kinds of surface are covered, each with its own frame type, because
//! they differ in what a caller can import from them:
//!
//! - [`PackedIoSurfaceFrame`] — one plane of packed 32-bit BGRA or RGBA
//!   pixels, as browser engines and compositors produce. It imports as one
//!   texture.
//! - [`Ycbcr420IoSurfaceFrame`] — biplanar 4:2:0 YCbCr, as cameras and video
//!   decoders produce: a full-resolution luma plane and a half-resolution
//!   plane of interleaved Cb/Cr pairs. Each [`Ycbcr420Plane`] imports as its
//!   own texture, at that plane's own extent:
//!
//! | `IOSurface` pixel format | Luma plane | Chroma plane | Device feature |
//! | --- | --- | --- | --- |
//! | `420v`, `420f` (8-bit) | `R8Unorm` | `Rg8Unorm` | — |
//! | `x420`, `xf20` (10-bit) | `R16Unorm` | `Rg16Unorm` | [`wgpu::Features::TEXTURE_FORMAT_16BIT_NORM`] |
//!
//! A 10-bit sample occupies the high ten bits of its 16-bit element, with the
//! low six bits zero, so a code value `v` reads back as
//! `(v << 6) / 65535` rather than `v / 1023`.
//!
//! Producers typically hand the surface over inside a callback and reclaim it
//! the moment that callback returns, so each frame type's `retain` takes a
//! reference on it first; the frame owns that reference until it is dropped.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CFRetained;
use objc2_io_surface::IOSurfaceRef;
use objc2_metal::{
    MTLDevice, MTLPixelFormat, MTLStorageMode, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};

use crate::YcbcrRange;

/// `kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange`.
const PIXEL_FORMAT_420V: u32 = u32::from_be_bytes(*b"420v");
/// `kCVPixelFormatType_420YpCbCr8BiPlanarFullRange`.
const PIXEL_FORMAT_420F: u32 = u32::from_be_bytes(*b"420f");
/// `kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange`.
const PIXEL_FORMAT_X420: u32 = u32::from_be_bytes(*b"x420");
/// `kCVPixelFormatType_420YpCbCr10BiPlanarFullRange`.
const PIXEL_FORMAT_XF20: u32 = u32::from_be_bytes(*b"xf20");

/// Channel order of a [`PackedIoSurfaceFrame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedFormat {
    /// 8-bit BGRA, one byte per channel.
    Bgra8,
    /// 8-bit RGBA, one byte per channel.
    Rgba8,
}

impl PackedFormat {
    /// The `wgpu` format the surface imports as.
    #[must_use]
    pub const fn texture_format(self) -> wgpu::TextureFormat {
        match self {
            Self::Bgra8 => wgpu::TextureFormat::Bgra8Unorm,
            Self::Rgba8 => wgpu::TextureFormat::Rgba8Unorm,
        }
    }

    const fn metal_format(self) -> MTLPixelFormat {
        match self {
            Self::Bgra8 => MTLPixelFormat::BGRA8Unorm,
            Self::Rgba8 => MTLPixelFormat::RGBA8Unorm,
        }
    }
}

/// Bits per sample of a [`Ycbcr420Format`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum YcbcrDepth {
    /// 8-bit samples, one byte each.
    Eight,
    /// 10-bit samples in the high bits of a 16-bit element.
    Ten,
}

/// One plane of a [`Ycbcr420IoSurfaceFrame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ycbcr420Plane {
    /// Plane 0: one luma sample per pixel.
    Luma,
    /// Plane 1: one interleaved Cb/Cr pair per 2×2 block of pixels.
    Chroma,
}

impl Ycbcr420Plane {
    /// The plane's index within the `IOSurface`.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Luma => 0,
            Self::Chroma => 1,
        }
    }
}

/// Pixel format of a biplanar 4:2:0 YCbCr surface: one of `420v`, `420f`,
/// `x420`, and `xf20`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ycbcr420Format {
    /// Bits per sample.
    pub depth: YcbcrDepth,
    /// The code range the samples span.
    pub range: YcbcrRange,
}

impl Ycbcr420Format {
    /// Maps an `IOSurface` / Core Video pixel format code onto the format it
    /// names, or `None` for any code that is not biplanar 4:2:0 YCbCr.
    #[must_use]
    pub const fn from_pixel_format(code: u32) -> Option<Self> {
        let (depth, range) = match code {
            PIXEL_FORMAT_420V => (YcbcrDepth::Eight, YcbcrRange::Video),
            PIXEL_FORMAT_420F => (YcbcrDepth::Eight, YcbcrRange::Full),
            PIXEL_FORMAT_X420 => (YcbcrDepth::Ten, YcbcrRange::Video),
            PIXEL_FORMAT_XF20 => (YcbcrDepth::Ten, YcbcrRange::Full),
            _ => return None,
        };
        Some(Self { depth, range })
    }

    /// The `IOSurface` / Core Video pixel format code of this format.
    #[must_use]
    pub const fn pixel_format(self) -> u32 {
        match (self.depth, self.range) {
            (YcbcrDepth::Eight, YcbcrRange::Video) => PIXEL_FORMAT_420V,
            (YcbcrDepth::Eight, YcbcrRange::Full) => PIXEL_FORMAT_420F,
            (YcbcrDepth::Ten, YcbcrRange::Video) => PIXEL_FORMAT_X420,
            (YcbcrDepth::Ten, YcbcrRange::Full) => PIXEL_FORMAT_XF20,
        }
    }

    /// The `wgpu` format `plane` imports as.
    #[must_use]
    pub const fn texture_format(self, plane: Ycbcr420Plane) -> wgpu::TextureFormat {
        match (self.depth, plane) {
            (YcbcrDepth::Eight, Ycbcr420Plane::Luma) => wgpu::TextureFormat::R8Unorm,
            (YcbcrDepth::Eight, Ycbcr420Plane::Chroma) => wgpu::TextureFormat::Rg8Unorm,
            (YcbcrDepth::Ten, Ycbcr420Plane::Luma) => wgpu::TextureFormat::R16Unorm,
            (YcbcrDepth::Ten, Ycbcr420Plane::Chroma) => wgpu::TextureFormat::Rg16Unorm,
        }
    }

    /// The device features importing a plane of this format needs.
    #[must_use]
    pub const fn required_features(self) -> wgpu::Features {
        match self.depth {
            YcbcrDepth::Eight => wgpu::Features::empty(),
            YcbcrDepth::Ten => wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
        }
    }

    const fn metal_format(self, plane: Ycbcr420Plane) -> MTLPixelFormat {
        match (self.depth, plane) {
            (YcbcrDepth::Eight, Ycbcr420Plane::Luma) => MTLPixelFormat::R8Unorm,
            (YcbcrDepth::Eight, Ycbcr420Plane::Chroma) => MTLPixelFormat::RG8Unorm,
            (YcbcrDepth::Ten, Ycbcr420Plane::Luma) => MTLPixelFormat::R16Unorm,
            (YcbcrDepth::Ten, Ycbcr420Plane::Chroma) => MTLPixelFormat::RG16Unorm,
        }
    }
}

/// A retained single-plane `IOSurface` of packed 32-bit pixels.
pub struct PackedIoSurfaceFrame {
    surface: CFRetained<IOSurfaceRef>,
    format: PackedFormat,
    width: u32,
    height: u32,
}

impl core::fmt::Debug for PackedIoSurfaceFrame {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PackedIoSurfaceFrame")
            .field("format", &self.format)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl PackedIoSurfaceFrame {
    /// Takes a reference on `surface`, so that it stays valid past the callback
    /// that handed it over.
    ///
    /// The extent is read from the surface, which is what it must be imported
    /// as; a producer that padded its allocation should narrow the *copy* it
    /// makes out of the imported texture. The channel order comes from the
    /// caller, who learns it from the producer (a browser engine reports it
    /// beside the surface), not from the surface's pixel format code.
    ///
    /// # Panics
    ///
    /// Panics when the surface is planar, or when its elements are not 4 bytes.
    ///
    /// # Safety
    ///
    /// `surface` must point at a live `IOSurface` for the duration of this
    /// call.
    #[must_use]
    pub unsafe fn retain(surface: NonNull<c_void>, format: PackedFormat) -> Self {
        // SAFETY: the caller contract makes `surface` a live `IOSurface`;
        // retaining it here is what keeps it valid afterwards.
        let surface = unsafe { CFRetained::retain(surface.cast::<IOSurfaceRef>()) };
        assert_eq!(
            surface.plane_count(),
            0,
            "a packed IOSurface frame needs a non-planar surface"
        );
        assert_eq!(
            surface.bytes_per_element(),
            4,
            "a packed IOSurface frame needs 4-byte elements for {format:?}"
        );
        let (width, height) = extent(surface.width(), surface.height());
        Self {
            surface,
            format,
            width,
            height,
        }
    }

    /// The surface's allocated pixel width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// The surface's allocated pixel height.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The channel order the surface imports as.
    #[must_use]
    pub const fn format(&self) -> PackedFormat {
        self.format
    }

    /// Imports the surface as a `COPY_SRC | TEXTURE_BINDING` texture on
    /// `device`.
    ///
    /// The texture aliases the surface's memory rather than owning a copy of
    /// it, so the producer may overwrite it as soon as it takes the surface
    /// back; copy out of, or finish sampling, the returned texture before that
    /// happens.
    ///
    /// # Panics
    ///
    /// Panics unless `device` is a Metal device.
    #[must_use]
    pub fn import(&self, device: &wgpu::Device) -> wgpu::Texture {
        import_plane(
            device,
            &self.surface,
            &PlaneImport {
                label: "wgpu_external_frame_io_surface",
                index: 0,
                format: self.format.texture_format(),
                metal_format: self.format.metal_format(),
                width: self.width,
                height: self.height,
            },
        )
    }
}

/// A retained biplanar 4:2:0 YCbCr `IOSurface`.
pub struct Ycbcr420IoSurfaceFrame {
    surface: CFRetained<IOSurfaceRef>,
    format: Ycbcr420Format,
    /// Width and height of each plane, indexed by [`Ycbcr420Plane::index`].
    extents: [(u32, u32); 2],
}

impl core::fmt::Debug for Ycbcr420IoSurfaceFrame {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Ycbcr420IoSurfaceFrame")
            .field("format", &self.format)
            .field("luma", &self.extents[0])
            .field("chroma", &self.extents[1])
            .finish_non_exhaustive()
    }
}

impl Ycbcr420IoSurfaceFrame {
    /// Takes a reference on `surface`, so that it stays valid past the callback
    /// that handed it over.
    ///
    /// The format and each plane's extent are read from the surface.
    ///
    /// # Panics
    ///
    /// Panics when the surface's pixel format is not one of `420v`, `420f`,
    /// `x420`, and `xf20`, or when it does not have two planes.
    ///
    /// # Safety
    ///
    /// `surface` must point at a live `IOSurface` for the duration of this
    /// call.
    #[must_use]
    pub unsafe fn retain(surface: NonNull<c_void>) -> Self {
        // SAFETY: the caller contract makes `surface` a live `IOSurface`;
        // retaining it here is what keeps it valid afterwards.
        let surface = unsafe { CFRetained::retain(surface.cast::<IOSurfaceRef>()) };
        let code = surface.pixel_format();
        let format = Ycbcr420Format::from_pixel_format(code).unwrap_or_else(|| {
            panic!(
                "IOSurface pixel format {:?} is not biplanar 4:2:0 YCbCr",
                String::from_utf8_lossy(&code.to_be_bytes())
            )
        });
        assert_eq!(
            surface.plane_count(),
            2,
            "a {format:?} IOSurface must have a luma and a chroma plane"
        );
        let extents = [Ycbcr420Plane::Luma, Ycbcr420Plane::Chroma].map(|plane| {
            extent(
                surface.width_of_plane(plane.index()),
                surface.height_of_plane(plane.index()),
            )
        });
        Self {
            surface,
            format,
            extents,
        }
    }

    /// The surface's pixel format.
    #[must_use]
    pub const fn format(&self) -> Ycbcr420Format {
        self.format
    }

    /// The pixel width of `plane`.
    #[must_use]
    pub const fn width(&self, plane: Ycbcr420Plane) -> u32 {
        self.extents[plane.index()].0
    }

    /// The pixel height of `plane`.
    #[must_use]
    pub const fn height(&self, plane: Ycbcr420Plane) -> u32 {
        self.extents[plane.index()].1
    }

    /// Imports `plane` as a `COPY_SRC | TEXTURE_BINDING` texture on `device`,
    /// at the plane's own extent and in
    /// [`Ycbcr420Format::texture_format`] for it.
    ///
    /// The texture aliases the surface's memory rather than owning a copy of
    /// it, so the producer may overwrite it as soon as it takes the surface
    /// back; copy out of, or finish sampling, the returned texture before that
    /// happens.
    ///
    /// # Panics
    ///
    /// Panics unless `device` is a Metal device with the
    /// [`Ycbcr420Format::required_features`] of the surface's format.
    #[must_use]
    pub fn import(&self, device: &wgpu::Device, plane: Ycbcr420Plane) -> wgpu::Texture {
        let required = self.format.required_features();
        assert!(
            device.features().contains(required),
            "importing a {:?} IOSurface needs a device created with {required:?}",
            self.format
        );
        let (width, height) = self.extents[plane.index()];
        import_plane(
            device,
            &self.surface,
            &PlaneImport {
                label: match plane {
                    Ycbcr420Plane::Luma => "wgpu_external_frame_io_surface_luma",
                    Ycbcr420Plane::Chroma => "wgpu_external_frame_io_surface_chroma",
                },
                index: plane.index(),
                format: self.format.texture_format(plane),
                metal_format: self.format.metal_format(plane),
                width,
                height,
            },
        )
    }
}

/// Converts an `IOSurface` extent to `wgpu`'s.
fn extent(width: usize, height: usize) -> (u32, u32) {
    (
        u32::try_from(width).expect("IOSurface width exceeds u32"),
        u32::try_from(height).expect("IOSurface height exceeds u32"),
    )
}

/// The storage mode a texture aliasing an `IOSurface` needs on `device`.
///
/// A Mac GPU without unified memory keeps a managed copy that Metal
/// synchronizes with the surface. iOS has no managed storage at all — the
/// simulator's GPU reports no unified memory, yet rejects `Managed` — so a
/// surface there is always shared.
#[cfg(target_os = "macos")]
fn storage_mode(device: &ProtocolObject<dyn MTLDevice>) -> MTLStorageMode {
    if device.hasUnifiedMemory() {
        MTLStorageMode::Shared
    } else {
        MTLStorageMode::Managed
    }
}

/// The storage mode a texture aliasing an `IOSurface` needs on `device`; see
/// the macOS version for why iOS has only one.
#[cfg(target_os = "ios")]
const fn storage_mode(_device: &ProtocolObject<dyn MTLDevice>) -> MTLStorageMode {
    MTLStorageMode::Shared
}

/// One plane of a surface, described in both APIs' terms.
struct PlaneImport {
    label: &'static str,
    index: usize,
    format: wgpu::TextureFormat,
    metal_format: MTLPixelFormat,
    width: u32,
    height: u32,
}

/// Imports plane `plane.index` of `surface` as a texture on `device`.
///
/// `plane` must describe that plane of `surface`: every frame type derives it
/// from the surface itself when the frame is retained.
fn import_plane(
    device: &wgpu::Device,
    surface: &IOSurfaceRef,
    plane: &PlaneImport,
) -> wgpu::Texture {
    let descriptor = wgpu::TextureDescriptor {
        label: Some(plane.label),
        size: wgpu::Extent3d {
            width: plane.width,
            height: plane.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: plane.format,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    let hal_texture = objc2::rc::autoreleasepool(|_| {
        // SAFETY: the handle is only borrowed to read the raw `MTLDevice`, and it is
        // not kept past this closure.
        let hal_device = unsafe {
            device
                .as_hal::<wgpu::hal::api::Metal>()
                .expect("IOSurface import requires a Metal device")
        };
        let raw_device = hal_device.raw_device();
        let metal_descriptor = MTLTextureDescriptor::new();
        // SAFETY: plain setters on a descriptor this scope just allocated and owns.
        unsafe {
            metal_descriptor
                .setWidth(usize::try_from(plane.width).expect("IOSurface width exceeds usize"));
            metal_descriptor
                .setHeight(usize::try_from(plane.height).expect("IOSurface height exceeds usize"));
        }
        metal_descriptor.setTextureType(MTLTextureType::Type2D);
        metal_descriptor.setPixelFormat(plane.metal_format);
        metal_descriptor.setUsage(MTLTextureUsage::ShaderRead);
        metal_descriptor.setStorageMode(storage_mode(raw_device));
        let texture = raw_device
            .newTextureWithDescriptor_iosurface_plane(&metal_descriptor, surface, plane.index)
            .unwrap_or_else(|| {
                panic!(
                    "Metal rejected importing IOSurface plane {} as {:?}",
                    plane.index, plane.format
                )
            });
        // SAFETY: the texture was created from `metal_descriptor` immediately above,
        // so the format, type, mip and layer counts repeated here match it, and
        // ownership of the `MTLTexture` transfers to the returned hal texture.
        unsafe {
            <wgpu::hal::api::Metal as wgpu::hal::Api>::Device::texture_from_raw(
                texture,
                descriptor.format,
                MTLTextureType::Type2D,
                1,
                1,
                wgpu::hal::CopyExtent {
                    width: plane.width,
                    height: plane.height,
                    depth: 1,
                },
                None,
            )
        }
    });
    // SAFETY: `hal_texture` was built for this device with `descriptor`'s format and
    // size, and is moved into the wgpu texture that now owns it. The surface's
    // contents are live, and `COPY_SRC` is one of the texture's usages, so it is a
    // state the texture may start its first barrier in.
    unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Metal>(
            hal_texture,
            &descriptor,
            wgpu::TextureUses::COPY_SRC,
        )
    }
}
