//! On-device tests of `AHardwareBuffer` import.
//!
//! Each test allocates a hardware buffer, writes a known pattern into it from
//! the CPU, imports it, and reads it back through a compute shader that samples
//! the imported texture, so what is compared is what the GPU sees. One test
//! hands the import a sync file a GPU submission signals, standing in for a
//! producer's acquire fence.
#![cfg(target_os = "android")]

use std::os::fd::{FromRawFd as _, OwnedFd};
use std::sync::mpsc;

use ash::vk;

use ndk::hardware_buffer::HardwareBufferRef;

use wgpu_external_frame::ahardware_buffer::{
    HardwareBuffer, HardwareBufferDesc, HardwareBufferFormat, HardwareBufferFrame,
    HardwareBufferImportError, HardwareBufferImporter, HardwareBufferLease, HardwareBufferUsage,
    request_device,
};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    importer: HardwareBufferImporter,
    pipeline: wgpu::ComputePipeline,
    spin_pipeline: wgpu::ComputePipeline,
}

impl Gpu {
    fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("the device has no Vulkan adapter");
        assert!(
            adapter
                .features()
                .contains(wgpu::Features::TEXTURE_FORMAT_NV12),
            "the Vulkan adapter does not offer TEXTURE_FORMAT_NV12"
        );
        let (device, queue) = request_device(
            &adapter,
            &wgpu::DeviceDescriptor {
                label: Some("ahardware_buffer_test"),
                required_features: wgpu::Features::TEXTURE_FORMAT_NV12,
                ..wgpu::DeviceDescriptor::default()
            },
        )
        .expect("failed to open a device that imports hardware buffers");
        let importer = HardwareBufferImporter::new(&device, &queue);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ahardware_buffer_read_texels"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ahardware_buffer.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ahardware_buffer_read_texels"),
            layout: None,
            module: &module,
            entry_point: Some("read_texels"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let spin_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ahardware_buffer_spin"),
            source: wgpu::ShaderSource::Wgsl(include_str!("spin.wgsl").into()),
        });
        let spin_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ahardware_buffer_spin"),
            layout: None,
            module: &spin_module,
            entry_point: Some("spin"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            device,
            queue,
            importer,
            pipeline,
            spin_pipeline,
        }
    }

    /// Submits a compute dispatch that keeps the GPU busy for a while.
    fn spin(&self) {
        const INVOCATIONS: u32 = 64 * 64;
        let sink = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ahardware_buffer_spin"),
            size: u64::from(INVOCATIONS) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ahardware_buffer_spin"),
            layout: &self.spin_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sink.as_entire_binding(),
            }],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.spin_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(INVOCATIONS / 64, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
    }

    /// Samples every texel of `view` (`width` × `height`) on the GPU and
    /// returns them as packed RGBA8 values, red in the low byte.
    fn read_texels(&self, view: &wgpu::TextureView, width: u32, height: u32) -> Vec<u32> {
        let size = u64::from(width * height) * 4;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ahardware_buffer_texels"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ahardware_buffer_texels"),
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        let readback = self.readback_buffer(size);
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
        self.queue.submit([encoder.finish()]);
        self.map(&readback)
    }

    /// Copies every texel of an RGBA8 `texture` into a buffer and returns them
    /// packed as [`Self::read_texels`] does.
    fn copy_texels(&self, texture: &wgpu::Texture) -> Vec<u32> {
        let size = texture.size();
        let bytes_per_row = size.width * 4;
        assert_eq!(
            bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
            0,
            "the test extent must copy without row padding"
        );
        let readback = self.readback_buffer(u64::from(bytes_per_row * size.height));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(size.height),
                },
            },
            size,
        );
        self.queue.submit([encoder.finish()]);
        self.map(&readback)
    }

    fn readback_buffer(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ahardware_buffer_readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn map(&self, buffer: &wgpu::Buffer) -> Vec<u32> {
        let slice = buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("the map result receiver is alive");
        });
        self.wait();
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

    fn wait(&self) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("waiting for the GPU failed");
    }
}

#[derive(Debug, PartialEq, Eq)]
enum LeaseEvent {
    Presented,
    Released,
}

#[derive(Debug)]
struct RecordingLease(mpsc::Sender<LeaseEvent>);

impl HardwareBufferLease for RecordingLease {
    fn presented(&mut self) {
        self.0
            .send(LeaseEvent::Presented)
            .expect("the lease event receiver is alive");
    }

    fn release(self: Box<Self>) {
        self.0
            .send(LeaseEvent::Released)
            .expect("the lease event receiver is alive");
    }
}

fn allocate(format: HardwareBufferFormat) -> HardwareBufferRef {
    HardwareBuffer::allocate(HardwareBufferDesc {
        width: WIDTH,
        height: HEIGHT,
        layers: 1,
        format,
        usage: HardwareBufferUsage::GPU_SAMPLED_IMAGE | HardwareBufferUsage::CPU_WRITE_OFTEN,
        stride: 0,
    })
    .unwrap_or_else(|error| panic!("failed to allocate a {format:?} hardware buffer: {error}"))
}

fn frame(
    buffer: &HardwareBuffer,
    fence: Option<OwnedFd>,
    lease: RecordingLease,
) -> HardwareBufferFrame {
    HardwareBufferFrame::new(buffer, fence).with_lease(Box::new(lease))
}

fn rgba_texel(x: u32, y: u32) -> [u8; 4] {
    [
        u8::try_from(x * 4).expect("fits"),
        u8::try_from(y * 5).expect("fits"),
        u8::try_from((x + y) * 2).expect("fits"),
        u8::try_from(255 - x).expect("fits"),
    ]
}

fn luma(x: u32, y: u32) -> u8 {
    u8::try_from((x * 3 + y * 2) % 256).expect("fits")
}

fn chroma(x: u32, y: u32) -> (u8, u8) {
    (
        u8::try_from(16 + x * 4 + y).expect("fits"),
        u8::try_from(240 - x * 3 - y * 2).expect("fits"),
    )
}

const fn pack(texel: [u8; 4]) -> u32 {
    u32::from_le_bytes(texel)
}

/// Writes [`rgba_texel`] into every pixel of an `R8G8B8A8_UNORM` buffer and
/// unlocks it synchronously.
fn write_rgba(buffer: &HardwareBuffer) {
    let stride = buffer.describe().stride;
    let pixels = buffer
        .lock(HardwareBufferUsage::CPU_WRITE_OFTEN, None, None)
        .expect("failed to lock the RGBA buffer")
        .cast::<u8>();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let offset = usize::try_from((y * stride + x) * 4).expect("fits usize");
            // SAFETY: the lock maps `stride` × `HEIGHT` four-byte pixels, and
            // `x < WIDTH <= stride`, `y < HEIGHT`, so the four bytes written
            // lie inside the mapping, which stays valid until the unlock.
            unsafe {
                pixels
                    .add(offset)
                    .copy_from_nonoverlapping(rgba_texel(x, y).as_ptr(), 4);
            }
        }
    }
    buffer.unlock().expect("failed to unlock the RGBA buffer");
}

fn expected_rgba() -> Vec<u32> {
    (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).map(move |x| pack(rgba_texel(x, y))))
        .collect()
}

#[test]
fn rgba_buffer_imports_with_its_texels() {
    let gpu = Gpu::new();
    let buffer = allocate(HardwareBufferFormat::R8G8B8A8_UNORM);
    write_rgba(&buffer);
    let (events, received) = mpsc::channel();
    let texture = gpu
        .importer
        .import(frame(&buffer, None, RecordingLease(events)))
        .expect("an RGBA buffer imports");
    assert_eq!(texture.format(), wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(received.try_recv(), Ok(LeaseEvent::Presented));

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    assert_eq!(
        gpu.read_texels(&view, WIDTH, HEIGHT),
        expected_rgba(),
        "sampled texels"
    );
    assert_eq!(gpu.copy_texels(&texture), expected_rgba(), "copied texels");
    assert!(
        received.try_recv().is_err(),
        "the lease must not be released while the texture is alive"
    );

    drop(view);
    drop(texture);
    gpu.wait();
    assert_eq!(received.try_recv(), Ok(LeaseEvent::Released));
}

/// A binary semaphore the GPU signals after a long compute dispatch, exported
/// as the sync file a producer would hand over with its buffer.
struct ProducerFence {
    device: ash::Device,
    semaphore: vk::Semaphore,
}

impl ProducerFence {
    fn signal(gpu: &Gpu) -> (Self, OwnedFd) {
        // SAFETY: the test device is a Vulkan device; the raw device is used
        // only to create, export and destroy a semaphore of the test's own.
        let hal_device = unsafe { gpu.device.as_hal::<wgpu::hal::api::Vulkan>() }
            .expect("the test device is Vulkan");
        let device = hal_device.raw_device().clone();
        let mut export = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut export);
        // SAFETY: a binary semaphore exportable as a sync file; the device
        // enables `VK_KHR_external_semaphore_fd` through `request_device`.
        let semaphore = unsafe { device.create_semaphore(&create_info, None) }
            .expect("failed to create the producer semaphore");
        let loader = ash::khr::external_semaphore_fd::Device::new(
            hal_device.shared_instance().raw_instance(),
            &device,
        );
        drop(hal_device);
        {
            // SAFETY: the test queue is Vulkan; the guard only stages a signal
            // of the semaphore on the next submission, which is the one below.
            let hal_queue = unsafe { gpu.queue.as_hal::<wgpu::hal::api::Vulkan>() }
                .expect("the test queue is Vulkan");
            hal_queue.add_signal_semaphore(semaphore, None);
        }
        gpu.spin();
        let get_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        // SAFETY: the semaphore's signal was just submitted, which a sync-file
        // export requires, and the extension is enabled.
        let fd = unsafe { loader.get_semaphore_fd(&get_info) }
            .expect("failed to export the producer semaphore");
        assert!(
            fd >= 0,
            "the spin dispatch finished before its semaphore was exported, so there is no \
             pending fence to test with"
        );
        // SAFETY: a successful export hands a new sync file to the caller.
        let fence = unsafe { OwnedFd::from_raw_fd(fd) };
        (Self { device, semaphore }, fence)
    }
}

impl Drop for ProducerFence {
    fn drop(&mut self) {
        // SAFETY: the test waits for the device to go idle before dropping
        // this, so the submission that signalled the semaphore has completed.
        unsafe { self.device.destroy_semaphore(self.semaphore, None) };
    }
}

/// The import takes a sync file that is still pending and turns it into the
/// semaphore its submission waits on. `wgpu` orders submissions on one queue,
/// so this proves the fence is consumed and the import completes with the
/// right contents, not that the wait alone orders the two.
#[test]
fn import_consumes_a_pending_acquire_fence() {
    let gpu = Gpu::new();
    let buffer = allocate(HardwareBufferFormat::R8G8B8A8_UNORM);
    write_rgba(&buffer);
    let (producer, fence) = ProducerFence::signal(&gpu);
    let (events, received) = mpsc::channel();
    let texture = gpu
        .importer
        .import(frame(&buffer, Some(fence), RecordingLease(events)))
        .expect("an RGBA buffer imports with an acquire fence");
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    assert_eq!(gpu.read_texels(&view, WIDTH, HEIGHT), expected_rgba());

    drop(view);
    drop(texture);
    gpu.wait();
    drop(producer);
    assert_eq!(
        received.try_iter().collect::<Vec<_>>(),
        [LeaseEvent::Presented, LeaseEvent::Released]
    );
}

#[test]
fn ycbcr_420_buffer_imports_as_nv12_or_reports_its_external_format() {
    let gpu = Gpu::new();
    let buffer = allocate(HardwareBufferFormat::Y8Cb8Cr8_420);
    let planes: Vec<_> = buffer
        .lock_planes(HardwareBufferUsage::CPU_WRITE_OFTEN, None, None)
        .expect("failed to lock the YCbCr buffer's planes")
        .collect();
    assert_eq!(
        planes.len(),
        3,
        "a YCbCr buffer locks as Y, Cb and Cr planes"
    );
    let write = |plane: usize, x: u32, y: u32, value: u8| {
        let plane = planes[plane];
        let offset = usize::try_from(y * plane.bytes_per_stride + x * plane.bytes_per_pixel)
            .expect("fits usize");
        // SAFETY: `lock_planes` maps each plane with the row and pixel strides
        // it reports, and every caller stays within that plane's extent — the
        // full extent for Y, half of it in each direction for Cb and Cr — so
        // the byte written lies inside the mapping, valid until the unlock.
        unsafe { plane.virtual_address.cast::<u8>().add(offset).write(value) };
    };
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            write(0, x, y, luma(x, y));
        }
    }
    for y in 0..HEIGHT / 2 {
        for x in 0..WIDTH / 2 {
            let (cb, cr) = chroma(x, y);
            write(1, x, y, cb);
            write(2, x, y, cr);
        }
    }
    buffer.unlock().expect("failed to unlock the YCbCr buffer");
    let (events, received) = mpsc::channel();
    let texture = match gpu
        .importer
        .import(frame(&buffer, None, RecordingLease(events)))
    {
        Ok(texture) => texture,
        Err(
            error @ HardwareBufferImportError::ExternalFormat {
                format,
                external_format,
            },
        ) => {
            assert_eq!(format, HardwareBufferFormat::Y8Cb8Cr8_420);
            assert_ne!(external_format, 0, "Vulkan external formats are never zero");
            assert!(
                error.to_string().contains("Y8Cb8Cr8_420"),
                "the rejection names the format: {error}"
            );
            assert_eq!(received.try_recv(), Err(mpsc::TryRecvError::Disconnected));
            return;
        }
        Err(error) => panic!("unexpected rejection of a Y8Cb8Cr8_420 buffer: {error}"),
    };
    assert_eq!(texture.format(), wgpu::TextureFormat::NV12);
    assert_eq!(received.try_recv(), Ok(LeaseEvent::Presented));

    let luma_view = texture.create_view(&wgpu::TextureViewDescriptor {
        format: Some(wgpu::TextureFormat::R8Unorm),
        aspect: wgpu::TextureAspect::Plane0,
        ..wgpu::TextureViewDescriptor::default()
    });
    let chroma_view = texture.create_view(&wgpu::TextureViewDescriptor {
        format: Some(wgpu::TextureFormat::Rg8Unorm),
        aspect: wgpu::TextureAspect::Plane1,
        ..wgpu::TextureViewDescriptor::default()
    });
    let expected_luma: Vec<u32> = (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).map(move |x| pack([luma(x, y), 0, 0, 255])))
        .collect();
    let expected_chroma: Vec<u32> = (0..HEIGHT / 2)
        .flat_map(|y| {
            (0..WIDTH / 2).map(move |x| {
                let (cb, cr) = chroma(x, y);
                pack([cb, cr, 0, 255])
            })
        })
        .collect();
    assert_eq!(
        gpu.read_texels(&luma_view, WIDTH, HEIGHT),
        expected_luma,
        "luma plane"
    );
    assert_eq!(
        gpu.read_texels(&chroma_view, WIDTH / 2, HEIGHT / 2),
        expected_chroma,
        "chroma plane"
    );

    drop(luma_view);
    drop(chroma_view);
    drop(texture);
    gpu.wait();
    assert_eq!(received.try_recv(), Ok(LeaseEvent::Released));
}

#[test]
fn rejected_buffer_drops_its_lease() {
    let gpu = Gpu::new();
    let buffer = HardwareBuffer::allocate(HardwareBufferDesc {
        width: WIDTH,
        height: HEIGHT,
        layers: 1,
        format: HardwareBufferFormat::R8G8B8A8_UNORM,
        usage: HardwareBufferUsage::CPU_WRITE_OFTEN | HardwareBufferUsage::CPU_READ_OFTEN,
        stride: 0,
    })
    .expect("failed to allocate a CPU-only hardware buffer");
    let (events, received) = mpsc::channel();
    let error = gpu
        .importer
        .import(frame(&buffer, None, RecordingLease(events)))
        .expect_err("a buffer without GPU_SAMPLED_IMAGE is rejected");
    assert!(
        matches!(error, HardwareBufferImportError::NotGpuSampled(_)),
        "unexpected rejection: {error}"
    );
    // The lease was dropped with the frame, never presented or released.
    assert_eq!(received.try_recv(), Err(mpsc::TryRecvError::Disconnected));
}
