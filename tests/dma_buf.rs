//! End-to-end tests of Vulkan and GLES DMA-BUF import.
//!
//! Each test allocates a DMA-BUF through `udmabuf` — a plain `memfd` exported
//! as a DMA-BUF, so no DRM device is needed — writes a known pattern into it
//! from the CPU, imports it on a `wgpu` device, and checks the texels the GPU
//! ends up holding.
//!
//! The tests assert the environment they need up front: `/dev/udmabuf`, an
//! adapter for the backend under test, and the backend's own DMA-BUF import
//! support — the `VK_KHR_external_memory_fd`,
//! `VK_EXT_external_memory_dma_buf`, and `VK_EXT_image_drm_format_modifier`
//! device extensions on Vulkan, `EGL_EXT_image_dma_buf_import` on EGL/GLES.
//! Any of those missing fails the test rather than skipping it.
#![cfg(target_os = "linux")]

use std::io;
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::sync::mpsc;

use wgpu_external_frame::dma_buf::{
    DmaBufFormat, DmaBufFrame, DmaBufImporter, DmaBufLease, DmaBufPlane,
};

/// `DRM_FORMAT_MOD_LINEAR` from `drm_fourcc.h`: the uncompressed linear
/// layout a `udmabuf` buffer always has.
const DRM_FORMAT_MOD_LINEAR: u64 = 0;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
/// The frame's row pitch. `u32` throughout because `DmaBufPlane::stride` is.
const STRIDE: u32 = WIDTH * 4;

/// Mirrors `struct udmabuf_create` in `linux/udmabuf.h`.
#[repr(C)]
struct UdmabufCreate {
    memfd: u32,
    flags: u32,
    offset: u64,
    size: u64,
}

const UDMABUF_FLAGS_CLOEXEC: u32 = 0x01;
// `_IOW('u', 0x42, struct udmabuf_create)` for a 24-byte struct.
const UDMABUF_CREATE: libc::c_ulong = 0x4018_7542;

/// A CPU-writable DMA-BUF: a `memfd` exported through `udmabuf`.
struct Udmabuf {
    /// The memory the buffer shares; writing it writes the buffer's pixels.
    /// The kernel references it from the DMA-BUF for as long as `fd` lives.
    _memfd: OwnedFd,
    /// The DMA-BUF descriptor, duplicated into each frame built from this.
    fd: OwnedFd,
    /// Byte size of the allocation, always a whole number of pages.
    size: usize,
}

impl Udmabuf {
    /// Allocates `size` bytes, rounded up to a whole page, as a DMA-BUF.
    ///
    /// `udmabuf` needs no GPU: it wraps an existing `memfd` in a DMA-BUF, so
    /// the same mapping serves CPU writes and the importer's reads.
    fn allocate(size: usize) -> Self {
        // SAFETY: `sysconf` is a read-only query; `_SC_PAGESIZE` is a valid
        // name for it on every Linux system.
        let page_size = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .expect("the system always reports a page size");
        let size = size.next_multiple_of(page_size);
        let device = std::fs::File::options()
            .read(true)
            .write(true)
            .open("/dev/udmabuf")
            .expect("allocating a DMA-BUF requires /dev/udmabuf");
        // SAFETY: `memfd_create` takes a NUL-terminated name, which a `c""`
        // literal supplies, and `MFD_CLOEXEC | MFD_ALLOW_SEALING` is a valid
        // flag set — `udmabuf` rejects memfds it cannot seal.
        let memfd = unsafe {
            libc::memfd_create(
                c"wgpu_external_frame_dma_buf".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        assert!(
            memfd >= 0,
            "memfd_create failed: {}",
            io::Error::last_os_error()
        );
        // SAFETY: `memfd` was just returned open by `memfd_create`.
        let memfd = unsafe { OwnedFd::from_raw_fd(memfd) };
        let size_t = libc::off_t::try_from(size).expect("the buffer size fits off_t");
        assert_eq!(
            // SAFETY: `ftruncate` sizes the `memfd` this function just
            // created and still owns; `size_t` is non-negative.
            unsafe { libc::ftruncate(memfd.as_raw_fd(), size_t) },
            0,
            "ftruncate failed: {}",
            io::Error::last_os_error()
        );
        // `udmabuf` wants `F_SEAL_SHRINK` on the backing `memfd` (and rejects
        // `F_SEAL_WRITE`); sealing after `ftruncate` keeps the size fixed.
        assert_ne!(
            // SAFETY: `fcntl` on a `memfd` created with `MFD_ALLOW_SEALING`
            // adds the seals given; `memfd` is open and still owned here.
            unsafe { libc::fcntl(memfd.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_SHRINK,) },
            -1,
            "sealing the memfd failed: {}",
            io::Error::last_os_error()
        );
        let create = UdmabufCreate {
            memfd: u32::try_from(memfd.as_raw_fd()).expect("the memfd fits u32"),
            flags: UDMABUF_FLAGS_CLOEXEC,
            offset: 0,
            size: u64::try_from(size).expect("the buffer size fits u64"),
        };
        // SAFETY: `UDMABUF_CREATE` reads the `udmabuf_create` struct passed
        // by pointer — `create` is a fully initialized local that outlives
        // the call — and returns a new descriptor or -1.
        let fd = unsafe { libc::ioctl(device.as_raw_fd(), UDMABUF_CREATE, &create) };
        assert!(
            fd >= 0,
            "UDMABUF_CREATE failed: {}",
            io::Error::last_os_error()
        );
        Self {
            _memfd: memfd,
            // SAFETY: `fd` was just returned open by the ioctl.
            fd: unsafe { OwnedFd::from_raw_fd(fd) },
            size,
        }
    }

    /// Writes `texel(x, y)` into every pixel of the buffer, at `stride` bytes
    /// per row.
    fn fill(&self, stride: usize, texel: impl Fn(u32, u32) -> [u8; 4]) {
        // SAFETY: `mmap` maps `self.size` bytes of `self.fd`, an open
        // `udmabuf` descriptor whose mapping the kernel provides, shared so
        // the writes land in the buffer itself.
        let map = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                self.size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.fd.as_raw_fd(),
                0,
            )
        };
        assert_ne!(
            map,
            libc::MAP_FAILED,
            "mmap of the DMA-BUF failed: {}",
            io::Error::last_os_error()
        );
        for y in 0..self.size / stride {
            for x in 0..stride / 4 {
                let offset = y * stride + x * 4;
                // SAFETY: `map` covers `self.size` bytes and `offset` stays
                // under `self.size`, as the loop bounds ensure.
                unsafe {
                    map.cast::<u8>().add(offset).copy_from_nonoverlapping(
                        texel(
                            u32::try_from(x).expect("the coordinate fits u32"),
                            u32::try_from(y).expect("the coordinate fits u32"),
                        )
                        .as_ptr(),
                        4,
                    );
                }
            }
        }
        assert_eq!(
            // SAFETY: `map` is the live mapping of `self.size` bytes from
            // above, unmapped exactly once here after the writes it served
            // are done.
            unsafe { libc::munmap(map, self.size) },
            0,
            "munmap of the DMA-BUF failed: {}",
            io::Error::last_os_error()
        );
    }

    /// A frame presenting `buffer` as one linear RGBA plane.
    fn frame(&self, lease: Box<dyn DmaBufLease>) -> DmaBufFrame {
        // SAFETY: `buffer.fd` is open; `dup` returns a fresh descriptor or -1.
        let fd = unsafe { libc::dup(self.fd.as_raw_fd()) };
        assert!(
            fd >= 0,
            "dup of the DMA-BUF descriptor failed: {}",
            io::Error::last_os_error()
        );
        // SAFETY: `fd` was just returned open by `dup`.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        DmaBufFrame::new(
            WIDTH,
            HEIGHT,
            DmaBufFormat::Rgba8,
            DRM_FORMAT_MOD_LINEAR,
            vec![DmaBufPlane {
                fd,
                offset: 0,
                stride: STRIDE,
            }],
            None,
        )
        .with_lease(lease)
    }
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    importer: DmaBufImporter,
    pipeline: wgpu::ComputePipeline,
}

impl Gpu {
    fn new(backends: wgpu::Backends) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            force_fallback_adapter: true,
            ..wgpu::RequestAdapterOptions::default()
        }))
        .expect("the tests require an adapter");
        match adapter.get_info().backend {
            wgpu::Backend::Vulkan => assert_vulkan_import_supported(&adapter),
            wgpu::Backend::Gl => assert_gles_import_supported(&adapter),
            backend => panic!("the tests require a Vulkan or GLES adapter, received {backend:?}"),
        }
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("failed to open the device");
        let importer = DmaBufImporter::new(&device, &queue, &adapter);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dma_buf_read_texels"),
            source: wgpu::ShaderSource::Wgsl(include_str!("dma_buf.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dma_buf_read_texels"),
            layout: None,
            module: &module,
            entry_point: Some("read_texels"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            device,
            queue,
            importer,
            pipeline,
        }
    }

    /// Records a dispatch sampling every texel of `view` into `readback`
    /// through the storage buffer `output`. `readback` must be `MAP_READ`
    /// with room for `width * height` packed texels.
    fn record_read_texels(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        readback: &wgpu::Buffer,
    ) {
        let size = u64::from(width * height) * 4;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dma_buf_texels"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dma_buf_texels"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, readback, 0, size);
    }

    /// Samples every texel of `view` (`width` × `height`) on the GPU and
    /// returns them as packed RGBA8 values, red in the low byte.
    fn read_texels(&self, view: &wgpu::TextureView, width: u32, height: u32) -> Vec<u32> {
        let readback = self.readback_buffer(u64::from(width * height) * 4);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.record_read_texels(&mut encoder, view, width, height, &readback);
        let submission = self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("waiting for the GPU failed");
        self.map(&readback)
    }

    /// Maps `buffer` and reads it as packed little-endian texels.
    fn map(&self, buffer: &wgpu::Buffer) -> Vec<u32> {
        let slice = buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("the map result receiver is alive");
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("waiting for the GPU failed");
        receiver
            .recv()
            .expect("the map callback ran")
            .expect("failed to map the readback buffer");
        let texels = slice
            .get_mapped_range()
            .expect("the readback buffer is mapped")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect();
        buffer.unmap();
        texels
    }

    fn readback_buffer(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dma_buf_readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
enum LeaseEvent {
    Presented,
    Released,
}

#[derive(Debug)]
struct RecordingLease(mpsc::Sender<LeaseEvent>);

impl DmaBufLease for RecordingLease {
    fn presented(&mut self) {
        self.0
            .send(LeaseEvent::Presented)
            .expect("the lease event receiver is alive");
    }

    fn release(self: Box<Self>, _release_fence: Option<OwnedFd>) {
        self.0
            .send(LeaseEvent::Released)
            .expect("the lease event receiver is alive");
    }
}

/// Asserts the Vulkan environment the import needs: an adapter whose
/// physical device offers the three external-memory extensions `wgpu`
/// enables for `VK_EXT_external_memory_dma_buf` import.
fn assert_vulkan_import_supported(adapter: &wgpu::Adapter) {
    // The import's device extensions come from `wgpu`, which enables them
    // whenever the driver offers them; assert the offer so a device that
    // cannot run the tests fails here rather than inside the import.
    // SAFETY: `Adapter::as_hal` requires naming the adapter's real
    // backend; a non-Vulkan adapter yields `None` and panics. The guard
    // is only read.
    let hal_adapter = unsafe {
        adapter
            .as_hal::<wgpu::hal::api::Vulkan>()
            .expect("the Vulkan tests require a Vulkan adapter")
    };
    let capabilities = hal_adapter.physical_device_capabilities();
    for extension in [
        ash::khr::external_memory_fd::NAME,
        ash::ext::external_memory_dma_buf::NAME,
        ash::ext::image_drm_format_modifier::NAME,
    ] {
        assert!(
            capabilities.supports_extension(extension),
            "the Vulkan adapter cannot import DMA-BUFs: {extension:?} is not supported"
        );
    }
}

/// Asserts the GLES environment the import needs: a `Gl` adapter whose EGL
/// display offers `EGL_EXT_image_dma_buf_import`.
fn assert_gles_import_supported(adapter: &wgpu::Adapter) {
    type EglGetCurrentDisplay = unsafe extern "C" fn() -> *mut core::ffi::c_void;
    type EglQueryString = unsafe extern "C" fn(*mut core::ffi::c_void, i32) -> *const i8;
    const EGL_EXTENSIONS: i32 = 0x3055;
    // SAFETY: `Library::new` is unsafe because `dlopen` runs the library's
    // initializers; this is the system EGL runtime by its versioned SONAME,
    // which the GLES adapter already has loaded, so the handle is the
    // existing one and no new initializer runs.
    let egl = unsafe { libloading::Library::new("libEGL.so.1") }
        .expect("the GLES tests require libEGL.so.1");
    // SAFETY: the signatures are EGL's own for these names, and the resolved
    // pointers stay valid while `egl` is alive, which is this function.
    let get_current_display = unsafe {
        *egl.get::<EglGetCurrentDisplay>(b"eglGetCurrentDisplay\0")
            .expect("libEGL.so.1 must export eglGetCurrentDisplay")
    };
    // SAFETY: as above, for `eglQueryString`.
    let query_string = unsafe {
        *egl.get::<EglQueryString>(b"eglQueryString\0")
            .expect("libEGL.so.1 must export eglQueryString")
    };
    // `wgpu` binds its EGL context only while one of its own calls runs, so
    // take the adapter's context lock while reading `eglGetCurrentDisplay`
    // and `eglQueryString`.
    // SAFETY: `Adapter::as_hal` requires the adapter's real backend; a
    // non-GLES adapter yields `None` and panics. The guard is only read.
    let hal_adapter = unsafe {
        adapter
            .as_hal::<wgpu::hal::api::Gles>()
            .expect("the GLES tests require a GLES adapter")
    };
    let _context = hal_adapter.adapter_context().lock();
    // SAFETY: `get_current_display` reads this thread's current EGL binding,
    // which the lock above just made the adapter's; `display` is then a live
    // `EGLDisplay` for `query_string`, and the returned C string outlives the
    // read while the display stays open, which the adapter guarantees.
    let extensions = unsafe {
        let display = get_current_display();
        assert!(
            !display.is_null(),
            "the GLES adapter's EGL context must be current under its own lock"
        );
        core::ffi::CStr::from_ptr(query_string(display, EGL_EXTENSIONS))
            .to_str()
            .expect("the EGL extension string is ASCII")
            .to_owned()
    };
    assert!(
        extensions
            .split(' ')
            .any(|extension| extension == "EGL_EXT_image_dma_buf_import"),
        "the EGL display cannot import DMA-BUFs: EGL_EXT_image_dma_buf_import is missing"
    );
}

/// The pattern written into the buffer: a texel that differs between
/// neighbouring pixels, rows, and channels, so a wrong row pitch or channel
/// order cannot reproduce it.
fn texel(x: u32, y: u32) -> [u8; 4] {
    [
        u8::try_from(x * 4).expect("fits"),
        u8::try_from(y * 5).expect("fits"),
        u8::try_from((x + y) * 2).expect("fits"),
        u8::try_from(255 - x).expect("fits"),
    ]
}

fn expected_texels() -> Vec<u32> {
    (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).map(move |x| u32::from_le_bytes(texel(x, y))))
        .collect()
}

/// `copy_to_texture` must import the buffer's contents: a shader reading the
/// returned texture sees exactly the pattern the CPU wrote.
fn copy_to_texture_imports_the_frame_on(backends: wgpu::Backends) {
    let gpu = Gpu::new(backends);
    let buffer = Udmabuf::allocate(usize::try_from(STRIDE * HEIGHT).expect("fits usize"));
    buffer.fill(usize::try_from(STRIDE).expect("fits usize"), texel);
    let (events, received) = mpsc::channel();
    let texture = gpu
        .importer
        .copy_to_texture(buffer.frame(Box::new(RecordingLease(events))));
    assert_eq!(texture.format(), wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(
        received.try_iter().collect::<Vec<_>>(),
        [LeaseEvent::Presented, LeaseEvent::Released],
        "a synchronous copy presents and releases the frame before returning"
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampled = gpu.read_texels(&view, WIDTH, HEIGHT);
    assert_eq!(sampled, expected_texels(), "sampled texels");
}
/// `copy_into` defers the wait: the caller presents the frame, records its own
/// commands into the returned encoder, and submits it in one `Queue::submit`
/// behind the import's command buffers. The readback dispatch here is the
/// caller's stand-in for a frame's real work — if the import's buffers and
/// the caller's were wrongly mixed or misordered, this either panics at
/// encode time or reads back an uninitialized texture instead of the frame.
/// `expected_command_buffers` is 3 on Vulkan — acquire, copy, release — and
/// 0 on GLES, which copies at `copy_into` call time instead.
fn copy_into_orders_the_copy_before_caller_commands_on(
    backends: wgpu::Backends,
    expected_command_buffers: usize,
) {
    let gpu = Gpu::new(backends);
    let buffer = Udmabuf::allocate(usize::try_from(STRIDE * HEIGHT).expect("fits usize"));
    buffer.fill(usize::try_from(STRIDE).expect("fits usize"), texel);
    let (events, received) = mpsc::channel();
    let destination = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dma_buf_copy_into_destination"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let mut frame = buffer.frame(Box::new(RecordingLease(events)));
    let mut import = gpu.importer.copy_into(&mut frame, &destination);
    assert_eq!(
        import.command_buffers.len(),
        expected_command_buffers,
        "unexpected number of import command buffers"
    );
    frame.presented();
    let readback = gpu.readback_buffer(u64::from(WIDTH * HEIGHT) * 4);
    let view = destination.create_view(&wgpu::TextureViewDescriptor::default());
    gpu.record_read_texels(&mut import.encoder, &view, WIDTH, HEIGHT, &readback);
    let submission = gpu.queue.submit(
        import
            .command_buffers
            .into_iter()
            .chain(std::iter::once(import.encoder.finish())),
    );
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("the DMA-BUF copy failed");
    drop(import.guard);
    frame.release(None);
    assert_eq!(
        received.try_iter().collect::<Vec<_>>(),
        [LeaseEvent::Presented, LeaseEvent::Released]
    );
    assert_eq!(gpu.map(&readback), expected_texels(), "sampled texels");
}

#[test]
fn vulkan_copy_to_texture_imports_the_frame() {
    copy_to_texture_imports_the_frame_on(wgpu::Backends::VULKAN);
}

#[test]
fn vulkan_copy_into_orders_the_copy_before_caller_commands() {
    copy_into_orders_the_copy_before_caller_commands_on(wgpu::Backends::VULKAN, 3);
}

#[test]
fn gles_copy_to_texture_imports_the_frame() {
    copy_to_texture_imports_the_frame_on(wgpu::Backends::GL);
}

#[test]
fn gles_copy_into_orders_the_copy_before_caller_commands() {
    copy_into_orders_the_copy_before_caller_commands_on(wgpu::Backends::GL, 0);
}
