//! Imports `IOSurface`s the test allocates and fills itself, and checks what a
//! shader binding each imported plane reads back against what was written.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use std::ptr::NonNull;
use std::sync::mpsc;

use objc2_core_foundation::{CFDictionary, CFNumber, CFRetained, CFString, CFType};
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferCreate, CVPixelBufferGetIOSurface,
    kCVPixelBufferIOSurfacePropertiesKey, kCVPixelFormatType_32BGRA, kCVPixelFormatType_32RGBA,
    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
    kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
    kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
    kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, kCVReturnSuccess,
};
use objc2_io_surface::{
    IOSurfaceLockOptions, IOSurfaceRef, kIOSurfaceBytesPerElement, kIOSurfaceHeight,
    kIOSurfacePixelFormat, kIOSurfaceWidth,
};
use wgpu_external_frame::io_surface::{
    PackedFormat, PackedIoSurfaceFrame, Ycbcr420Format, Ycbcr420IoSurfaceFrame, Ycbcr420Plane,
    YcbcrDepth, YcbcrRange,
};

/// Odd on purpose: the chroma plane's extent then has to be rounded up, which
/// only reading it from the surface gets right.
const WIDTH: usize = 71;
const HEIGHT: usize = 45;

/// How the samples of one plane are laid out in memory.
#[derive(Debug, Clone, Copy)]
struct Samples {
    /// Channels per texel.
    channels: usize,
    /// Whether each sample is a 16-bit element holding a 10-bit code in its
    /// high bits, rather than one byte.
    ten_bit: bool,
}

impl Samples {
    const fn bytes(self) -> usize {
        if self.ten_bit { 2 } else { 1 }
    }

    /// The element stored for channel `channel` of texel (`x`, `y`): a pattern
    /// that differs between neighbouring texels, rows, and channels, so a
    /// wrong row pitch, plane, or channel order cannot reproduce it.
    fn stored(self, x: usize, y: usize, channel: usize) -> u16 {
        if self.ten_bit {
            let code = (x * 37 + y * 61 + channel * 509) % 1024;
            u16::try_from(code << 6).expect("a 10-bit code fits in 16 bits")
        } else {
            u16::try_from((x * 7 + y * 13 + channel * 101) % 256).expect("a byte fits in 16 bits")
        }
    }

    /// What a shader reads for a stored element.
    fn normalized(self, stored: u16) -> f32 {
        f32::from(stored) / self.max()
    }

    /// The largest element, which a shader reads as one.
    const fn max(self) -> f32 {
        if self.ten_bit { 65535.0 } else { 255.0 }
    }
}

fn gpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("IOSurface tests need a Metal adapter");
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("io_surface_test"),
        required_features: wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
        // The defaults exceed what the iOS simulator's GPU offers, and these
        // tests need nothing beyond what the adapter has.
        required_limits: adapter.limits(),
        ..wgpu::DeviceDescriptor::default()
    }))
    .expect("the Metal adapter refused the test device")
}

/// Allocates a non-planar surface of 4-byte pixels directly, the way browser
/// engines and compositors do.
fn packed_surface(pixel_format: u32) -> CFRetained<IOSurfaceRef> {
    let number = |value: usize| {
        CFNumber::new_isize(isize::try_from(value).expect("the property fits isize"))
    };
    let (width, height, bytes_per_element) = (number(WIDTH), number(HEIGHT), number(4));
    let pixel_format =
        CFNumber::new_i32(i32::try_from(pixel_format).expect("packed pixel format codes fit i32"));
    // SAFETY: IOSurface defines these keys as immutable `CFString`s that live
    // for the whole process.
    let keys = unsafe {
        [
            kIOSurfaceWidth,
            kIOSurfaceHeight,
            kIOSurfaceBytesPerElement,
            kIOSurfacePixelFormat,
        ]
    };
    let properties = CFDictionary::<CFString, CFNumber>::from_slices(
        &keys,
        &[&width, &height, &bytes_per_element, &pixel_format],
    );
    // SAFETY: `properties` is a live dictionary of the keys and value types
    // `IOSurfaceCreate` documents.
    unsafe { IOSurfaceRef::new(properties.as_opaque()) }.expect("IOSurfaceCreate failed")
}

/// Allocates an `IOSurface`-backed pixel buffer, the way capture and decode
/// pipelines do, and returns its surface.
fn pixel_buffer_surface(pixel_format: u32) -> CFRetained<IOSurfaceRef> {
    let surface_properties = CFDictionary::<CFString, CFType>::empty();
    // SAFETY: Core Video defines the key as an immutable `CFString` that lives
    // for the whole process.
    let key = unsafe { kCVPixelBufferIOSurfacePropertiesKey };
    let attributes =
        CFDictionary::<CFString, CFType>::from_slices(&[key], &[surface_properties.as_ref()]);
    let mut buffer: *mut CVPixelBuffer = std::ptr::null_mut();
    // SAFETY: `attributes` is a live dictionary of the shape Core Video
    // documents, and `buffer` is a local the call writes its result into.
    let status = unsafe {
        CVPixelBufferCreate(
            None,
            WIDTH,
            HEIGHT,
            pixel_format,
            Some(attributes.as_opaque()),
            NonNull::from(&mut buffer),
        )
    };
    assert_eq!(
        status,
        kCVReturnSuccess,
        "CVPixelBufferCreate failed for pixel format {:?}",
        String::from_utf8_lossy(&pixel_format.to_be_bytes())
    );
    let buffer = NonNull::new(buffer).expect("CVPixelBufferCreate succeeded with no buffer");
    // SAFETY: a successful `CVPixelBufferCreate` hands its caller a +1
    // reference, which this adopts.
    let buffer = unsafe { CFRetained::from_raw(buffer) };
    CVPixelBufferGetIOSurface(Some(&buffer)).expect("the pixel buffer has no IOSurface backing")
}

/// The part of a surface [`fill`] writes.
#[derive(Debug, Clone, Copy)]
enum Region {
    /// The whole of a non-planar surface.
    Whole,
    /// One plane of a planar surface.
    Plane(Ycbcr420Plane),
}

/// Writes [`Samples::stored`] into `region` of `surface`.
fn fill(surface: &IOSurfaceRef, region: Region, samples: Samples) {
    // SAFETY: a null seed pointer is allowed; the surface is locked for
    // writing only for the span of this function.
    let status = unsafe { surface.lock(IOSurfaceLockOptions::empty(), std::ptr::null_mut()) };
    assert_eq!(status, 0, "IOSurfaceLock failed");
    let (base, row_bytes, width, height) = match region {
        Region::Whole => (
            surface.base_address(),
            surface.bytes_per_row(),
            surface.width(),
            surface.height(),
        ),
        Region::Plane(plane) => (
            surface.base_address_of_plane(plane.index()),
            surface.bytes_per_row_of_plane(plane.index()),
            surface.width_of_plane(plane.index()),
            surface.height_of_plane(plane.index()),
        ),
    };
    assert!(
        width * samples.channels * samples.bytes() <= row_bytes,
        "the plane's rows are narrower than its samples"
    );
    let base = base.cast::<u8>();
    for y in 0..height {
        for x in 0..width {
            for channel in 0..samples.channels {
                let offset = y * row_bytes + (x * samples.channels + channel) * samples.bytes();
                let stored = samples.stored(x, y, channel);
                // SAFETY: the surface is locked, so its base address is mapped,
                // and `offset` lies inside row `y`, which the assertion above
                // shows holds every sample of the row.
                unsafe {
                    let element = base.add(offset);
                    if samples.ten_bit {
                        element
                            .cast::<[u8; 2]>()
                            .write_unaligned(stored.to_ne_bytes());
                    } else {
                        element.write(u8::try_from(stored).expect("an 8-bit sample"));
                    }
                }
            }
        }
    }
    // SAFETY: as for the lock above.
    let status = unsafe { surface.unlock(IOSurfaceLockOptions::empty(), std::ptr::null_mut()) };
    assert_eq!(status, 0, "IOSurfaceUnlock failed");
}

/// Reads every texel of `texture` through a shader binding of it.
fn sample(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<[f32; 4]> {
    let size = texture.size();
    let texels = u64::from(size.width) * u64::from(size.height);
    let bytes = texels * 16;
    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("io_surface_test_texels"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("io_surface_test_readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(wgpu::include_wgsl!("sample_plane.wgsl"));
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("io_surface_test_sample"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("io_surface_test_sample"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: storage.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("io_surface_test_sample"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(size.width.div_ceil(8), size.height.div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, bytes);
    queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("the test is waiting for the map");
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("the sampling submission failed");
    receiver
        .recv()
        .expect("the map callback never ran")
        .expect("mapping the readback buffer failed");
    let mapped = readback
        .slice(..)
        .get_mapped_range()
        .expect("the readback buffer is mapped");
    let (texels, rest) = mapped.as_chunks::<16>();
    assert!(rest.is_empty(), "the readback holds whole texels");
    texels
        .iter()
        .map(|texel| {
            let (channels, _) = texel.as_chunks::<4>();
            std::array::from_fn(|channel| f32::from_le_bytes(channels[channel]))
        })
        .collect()
}

/// Asserts that `texture` reads back as what [`fill`] wrote, where sampled
/// channel `i` comes from stored channel `swizzle[i]`.
fn assert_texels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    samples: Samples,
    swizzle: &[usize],
) {
    let size = texture.size();
    let width = usize::try_from(size.width).expect("width fits usize");
    let texels = sample(device, queue, texture);
    for (index, texel) in texels.iter().enumerate() {
        let (x, y) = (index % width, index / width);
        for (channel, &stored_channel) in swizzle.iter().enumerate() {
            let expected = samples.normalized(samples.stored(x, y, stored_channel));
            let read = texel[channel];
            assert!(
                (read - expected).abs() < 0.5 / samples.max(),
                "{:?} texel ({x}, {y}) channel {channel}: read {read}, wrote {expected}",
                texture.format()
            );
        }
    }
}

fn check_packed(pixel_format: u32, format: PackedFormat, swizzle: [usize; 4]) {
    let (device, queue) = gpu();
    let surface = packed_surface(pixel_format);
    let samples = Samples {
        channels: 4,
        ten_bit: false,
    };
    fill(&surface, Region::Whole, samples);
    // SAFETY: `surface` is a live IOSurface this test retains.
    let frame = unsafe { PackedIoSurfaceFrame::retain(NonNull::from(&*surface).cast(), format) };
    drop(surface);
    assert_eq!((frame.width(), frame.height()), (71, 45));
    let texture = frame.import(&device);
    assert_eq!(texture.format(), format.texture_format());
    assert_eq!((texture.width(), texture.height()), (71, 45));
    assert_texels(&device, &queue, &texture, samples, &swizzle);
}

fn check_ycbcr420(pixel_format: u32, expected: Ycbcr420Format) {
    let (device, queue) = gpu();
    let surface = pixel_buffer_surface(pixel_format);
    let ten_bit = expected.depth == YcbcrDepth::Ten;
    let luma = Samples {
        channels: 1,
        ten_bit,
    };
    let chroma = Samples {
        channels: 2,
        ten_bit,
    };
    fill(&surface, Region::Plane(Ycbcr420Plane::Luma), luma);
    fill(&surface, Region::Plane(Ycbcr420Plane::Chroma), chroma);
    // SAFETY: `surface` is a live IOSurface this test retains.
    let frame = unsafe { Ycbcr420IoSurfaceFrame::retain(NonNull::from(&*surface).cast()) };
    drop(surface);
    assert_eq!(frame.format(), expected);
    assert_eq!(frame.format().pixel_format(), pixel_format);
    for (plane, samples, extent) in [
        (Ycbcr420Plane::Luma, luma, (71, 45)),
        (Ycbcr420Plane::Chroma, chroma, (36, 23)),
    ] {
        assert_eq!((frame.width(plane), frame.height(plane)), extent);
        let texture = frame.import(&device, plane);
        assert_eq!(texture.format(), expected.texture_format(plane));
        assert_eq!((texture.width(), texture.height()), extent);
        assert_texels(
            &device,
            &queue,
            &texture,
            samples,
            &[0, 1][..samples.channels],
        );
    }
}

#[test]
fn bgra_surface_imports_its_pixels() {
    check_packed(kCVPixelFormatType_32BGRA, PackedFormat::Bgra8, [2, 1, 0, 3]);
}

#[test]
fn rgba_surface_imports_its_pixels() {
    check_packed(kCVPixelFormatType_32RGBA, PackedFormat::Rgba8, [0, 1, 2, 3]);
}

#[test]
fn ycbcr420_8bit_video_range_imports_both_planes() {
    check_ycbcr420(
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        Ycbcr420Format {
            depth: YcbcrDepth::Eight,
            range: YcbcrRange::Video,
        },
    );
}

#[test]
fn ycbcr420_8bit_full_range_imports_both_planes() {
    check_ycbcr420(
        kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        Ycbcr420Format {
            depth: YcbcrDepth::Eight,
            range: YcbcrRange::Full,
        },
    );
}

#[test]
fn ycbcr420_10bit_video_range_imports_both_planes() {
    check_ycbcr420(
        kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
        Ycbcr420Format {
            depth: YcbcrDepth::Ten,
            range: YcbcrRange::Video,
        },
    );
}

#[test]
fn ycbcr420_10bit_full_range_imports_both_planes() {
    check_ycbcr420(
        kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
        Ycbcr420Format {
            depth: YcbcrDepth::Ten,
            range: YcbcrRange::Full,
        },
    );
}
