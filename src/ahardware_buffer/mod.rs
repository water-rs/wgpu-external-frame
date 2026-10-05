//! Android `AHardwareBuffer` import.
//!
//! Android producers — `AImageReader` with GPU usage, `MediaCodec`,
//! `SurfaceControl` — hand out frames as `AHardwareBuffer`s. Vulkan imports
//! one through `VK_ANDROID_external_memory_android_hardware_buffer`, and `wgpu`
//! adopts the resulting image through its `hal` layer, so the returned
//! [`wgpu::Texture`] *is* the producer's buffer: no pixel is copied, by the CPU
//! or the GPU.
//!
//! # Device
//!
//! The import needs device extensions `wgpu` does not enable by itself
//! ([`DEVICE_EXTENSIONS`]). Open the device with [`request_device`], or add
//! the extensions in `wgpu-hal`'s device-creation callback when a renderer
//! opens its device itself.
//!
//! # Formats
//!
//! The format comes from the Vulkan driver's own reading of the buffer
//! (`vkGetAndroidHardwareBufferPropertiesANDROID`), not from the buffer's
//! description:
//!
//! | Vulkan format the driver reports | `wgpu` texture | Usages |
//! | --- | --- | --- |
//! | `R8G8B8A8_UNORM` (`R8G8B8A8_UNORM`, `R8G8B8X8_UNORM` buffers) | `Rgba8Unorm` | `TEXTURE_BINDING`, `COPY_SRC` |
//! | `G8_B8R8_2PLANE_420_UNORM` (`Y8Cb8Cr8_420` buffers, when the driver maps them) | `NV12` | `TEXTURE_BINDING` |
//!
//! An `NV12` texture is read through plane views: `TextureAspect::Plane0` as
//! `R8Unorm` for luma and `TextureAspect::Plane1` as `Rg8Unorm` for the
//! interleaved Cb/Cr pair, which needs `wgpu::Features::TEXTURE_FORMAT_NV12`
//! on the device.
//!
//! A buffer the driver can only describe with an implementation-defined
//! *external format* — Vulkan format `UNDEFINED`, typical of camera and video
//! buffers allocated with `AIMAGE_FORMAT_PRIVATE` — is rejected with
//! [`HardwareBufferImportError::ExternalFormat`]. Sampling such a buffer needs
//! a `VkSamplerYcbcrConversion` bound into the pipeline, which `wgpu` cannot
//! express; a producer that should be imported here has to be configured for a
//! format with a Vulkan equivalent instead.
//!
//! # Ownership
//!
//! A [`HardwareBufferFrame`] holds a reference on the buffer, the producer's
//! acquire fence, and its [`HardwareBufferLease`]. Importing consumes the
//! frame:
//!
//! 1. The lease is told the frame was
//!    [presented](HardwareBufferLease::presented), and the import submits the
//!    acquire of the buffer from the producer's (foreign) queue family, which
//!    waits on the acquire fence on the GPU.
//! 2. The texture can be sampled for as long as the caller keeps it.
//! 3. Once the texture is dropped and every submission that used it has
//!    completed, `wgpu` destroys it; that releases the Vulkan objects, the
//!    buffer reference, and finally the lease, which hands the buffer back to
//!    the producer.
//!
//! The texture is read-only — no usage lets the GPU write it — so the producer
//! gets back exactly what it wrote.

mod device;
mod frame;
mod vulkan;

pub use device::{DEVICE_EXTENSIONS, DeviceRequestError, request_device};
pub use frame::{HardwareBufferFrame, HardwareBufferLease};
pub use ndk::hardware_buffer::{HardwareBuffer, HardwareBufferDesc, HardwareBufferUsage};
pub use ndk::hardware_buffer_format::HardwareBufferFormat;

use ash::vk;
use frame::{BufferReference, FrameParts};
use sync_wrapper::SyncWrapper;
use vulkan::{ImageShape, ImportedImage};

/// Why a hardware buffer cannot be imported.
///
/// Every variant is a property of the buffer the producer handed over, so the
/// frame is dropped — releasing its lease — when an import is rejected.
#[derive(Debug, thiserror::Error)]
pub enum HardwareBufferImportError {
    /// The driver describes the buffer only through an implementation-defined
    /// external format, which needs a `VkSamplerYcbcrConversion` that `wgpu`
    /// cannot express.
    #[error(
        "the AHardwareBuffer format {format:?} has no Vulkan format on this device, only the \
         external format {external_format:#x}; sampling it needs a VkSamplerYcbcrConversion, \
         which wgpu cannot express"
    )]
    ExternalFormat {
        /// The buffer's own format.
        format: HardwareBufferFormat,
        /// The driver's implementation-defined format identifier.
        external_format: u64,
    },
    /// The buffer has a Vulkan format, but not one this import maps to a
    /// `wgpu` texture format.
    #[error(
        "the AHardwareBuffer format {format:?} imports as the Vulkan format {}, which this \
         import does not support",
        vk_format_name(*.vulkan_format)
    )]
    UnsupportedFormat {
        /// The buffer's own format.
        format: HardwareBufferFormat,
        /// The raw `VkFormat` value the driver reported.
        vulkan_format: i32,
    },
    /// The buffer was not allocated for GPU sampling.
    #[error("the AHardwareBuffer usage {0:?} lacks GPU_SAMPLED_IMAGE")]
    NotGpuSampled(HardwareBufferUsage),
    /// The buffer holds protected content, which only a protected device can
    /// read.
    #[error("the AHardwareBuffer holds protected content")]
    Protected,
    /// The buffer has more than one layer.
    #[error("the AHardwareBuffer has {0} layers; only single-layer buffers import as a 2D texture")]
    Layered(u32),
}

fn vk_format_name(raw: i32) -> String {
    format!("{:?}", vk::Format::from_raw(raw))
}

/// Imports Android hardware buffers into textures on one `wgpu` device.
#[derive(Debug)]
pub struct HardwareBufferImporter {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl HardwareBufferImporter {
    /// Creates an importer for `device` and the queue imports are submitted
    /// on.
    ///
    /// # Panics
    ///
    /// Panics unless `device` is a Vulkan device with every one of
    /// [`DEVICE_EXTENSIONS`] enabled.
    #[must_use]
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // SAFETY: `Device::as_hal` requires naming the device's real backend;
        // a non-Vulkan device yields `None` and panics here. The guard is only
        // read and dropped at the end of the statement.
        let hal_device = unsafe {
            device
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("AHardwareBuffer import requires a Vulkan device")
        };
        vulkan::validate_device(&hal_device);
        drop(hal_device);
        Self {
            device: device.clone(),
            queue: queue.clone(),
        }
    }

    /// Imports `frame` as a texture that aliases the buffer, and submits the
    /// acquire of the buffer from its producer.
    ///
    /// The returned texture is `TEXTURE_BINDING` (plus `COPY_SRC` for
    /// single-plane formats), of the format in the module-level table, at the
    /// buffer's full allocated extent. Submissions made after this returns
    /// observe the producer's writes: the import submission waits on the
    /// frame's acquire fence on the GPU.
    ///
    /// # Errors
    ///
    /// Returns an error when the buffer is not GPU-sampled, protected,
    /// layered, or of a format that has no `wgpu` equivalent — in particular an
    /// implementation-defined external format. The frame, and with it the
    /// lease, is dropped.
    ///
    /// # Panics
    ///
    /// Panics when the buffer is multi-planar YCbCr and the device lacks
    /// `wgpu::Features::TEXTURE_FORMAT_NV12`, when its extent or mip chain is
    /// one `wgpu` cannot address for its format (an odd-sized or mipmapped
    /// `NV12` buffer), or when Vulkan fails to import it.
    pub fn import(
        &self,
        frame: HardwareBufferFrame,
    ) -> Result<wgpu::Texture, HardwareBufferImportError> {
        let FrameParts {
            buffer,
            description,
            acquire_fence,
            mut lease,
        } = frame.into_parts();
        check_description(&description)?;
        // SAFETY: `Device::as_hal` requires naming the device's real backend,
        // which `new` asserted is Vulkan, and that the raw device is not used
        // to invalidate `wgpu`'s state. It is used only to create and query
        // objects of this import's own.
        let hal_device = unsafe {
            self.device
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("AHardwareBuffer import requires a Vulkan device")
        };
        let properties = vulkan::buffer_properties(&hal_device, buffer.buffer());
        let layout = TextureLayout::new(&description, &properties)?;
        let required = layout.format.required_features();
        assert!(
            self.device.features().contains(required),
            "importing an AHardwareBuffer of format {:?} as {:?} requires the device feature \
             {required:?}",
            description.format,
            layout.format,
        );
        let imported = vulkan::import_buffer(
            &hal_device,
            buffer.buffer(),
            &properties,
            &layout.image_shape(properties.format),
            acquire_fence,
        );
        let image = imported.image();
        let acquire_semaphore = imported.acquire_semaphore();
        let raw_device = hal_device.raw_device().clone();
        let queue_family_index = hal_device.queue_family_index();
        if let Some(lease) = lease.as_mut() {
            lease.presented();
        }
        let retirement = SyncWrapper::new(Retirement {
            imported,
            buffer,
            lease,
        });
        let drop_callback: wgpu::hal::DropCallback =
            Box::new(move || retirement.into_inner().retire());
        // SAFETY: `texture_from_raw` requires an image created to match the
        // descriptor, which it was: `image_shape` and `hal_descriptor` describe
        // the same extent, single layer, mip count and sample count, the image
        // has the Vulkan format `wgpu` maps the texture format to, and its
        // usages cover the descriptor's. `view_formats` is empty; the
        // multi-planar case is created mutable, as `wgpu` itself does for these
        // formats. With a drop callback the image stays this module's to
        // destroy, which the callback does through `Retirement`, and
        // `TextureMemory::External` leaves the memory to it as well.
        let hal_texture = unsafe {
            hal_device.texture_from_raw(
                image,
                &layout.hal_descriptor(),
                Some(drop_callback),
                wgpu::hal::vulkan::TextureMemory::External,
            )
        };
        drop(hal_device);
        // SAFETY: `hal_texture` was created on this device from a hal
        // descriptor that matches this one, and its memory holds the
        // producer's contents, so it counts as initialized. It is declared to
        // start in `RESOURCE`, which is the state the acquire barrier submitted
        // below leaves it in before any later submission can use it.
        let texture = unsafe {
            self.device
                .create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                    hal_texture,
                    &layout.descriptor(),
                    wgpu::TextureUses::RESOURCE,
                )
        };
        self.submit_acquire(
            &texture,
            image,
            &raw_device,
            queue_family_index,
            acquire_semaphore,
        );
        Ok(texture)
    }

    fn submit_acquire(
        &self,
        texture: &wgpu::Texture,
        image: vk::Image,
        raw_device: &ash::Device,
        queue_family_index: u32,
        acquire_semaphore: Option<vk::Semaphore>,
    ) {
        // `wgpu` forbids mixing raw and `wgpu` commands in one encoder, so the
        // acquire is recorded raw into one, and a second, recorded through
        // `wgpu`, carries the texture's use. Both go into one submission.
        let mut acquire = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu_external_frame_ahardware_buffer_acquire"),
            });
        // SAFETY: recording raw Vulkan into a `wgpu` encoder requires naming
        // its real backend, which `new` asserted is Vulkan, and leaving the
        // encoder in a state `wgpu` can finish. One pipeline barrier is
        // recorded, outside any render pass, allocating nothing, into an
        // encoder no `wgpu` command touches. `image` belongs to `texture`,
        // whose retirement this submission keeps alive, as described below.
        unsafe {
            acquire.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|encoder| {
                let encoder = encoder.expect("AHardwareBuffer command encoder is not Vulkan");
                vulkan::record_acquire(raw_device, encoder, image, queue_family_index);
            });
        }
        let mut track = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu_external_frame_ahardware_buffer_track"),
            });
        // A transition to the state the texture is already in records no
        // barrier, but it enters the texture in this submission's tracker, so
        // `wgpu` keeps the texture — and through its drop callback the image,
        // memory, and acquire semaphore — alive until the submission
        // completes, even if the caller drops it at once.
        track.transition_resources(
            core::iter::empty(),
            core::iter::once(wgpu::TextureTransition {
                texture,
                selector: None,
                state: wgpu::TextureUses::RESOURCE,
            }),
        );
        let command_buffers = [acquire.finish(), track.finish()];
        if let Some(semaphore) = acquire_semaphore {
            // SAFETY: `Queue::as_hal` requires naming the queue's real backend,
            // which is the device's, Vulkan. The guard only stages a wait on
            // the semaphore for the queue's next submission, which is the
            // `submit` right below unless another thread submits first — in
            // which case that earlier submission waits instead, and this one
            // still runs after it, as `wgpu` orders submissions on its queue.
            // The semaphore holds a pending sync-file payload and is destroyed
            // only after this submission completes.
            let hal_queue = unsafe {
                self.queue
                    .as_hal::<wgpu::hal::api::Vulkan>()
                    .expect("AHardwareBuffer import requires a Vulkan queue")
            };
            hal_queue.add_wait_semaphore(semaphore, None, vk::PipelineStageFlags::ALL_COMMANDS);
        }
        self.queue.submit(command_buffers);
    }
}

/// Rejects buffers whose description rules out a 2D sampled import.
const fn check_description(
    description: &HardwareBufferDesc,
) -> Result<(), HardwareBufferImportError> {
    if !description
        .usage
        .contains(HardwareBufferUsage::GPU_SAMPLED_IMAGE)
    {
        return Err(HardwareBufferImportError::NotGpuSampled(description.usage));
    }
    if description
        .usage
        .contains(HardwareBufferUsage::PROTECTED_CONTENT)
    {
        return Err(HardwareBufferImportError::Protected);
    }
    if description.layers != 1 {
        return Err(HardwareBufferImportError::Layered(description.layers));
    }
    Ok(())
}

/// The `wgpu` texture a buffer imports as.
struct TextureLayout {
    format: wgpu::TextureFormat,
    size: wgpu::Extent3d,
    mip_level_count: u32,
    usage: wgpu::TextureUsages,
}

impl TextureLayout {
    const LABEL: Option<&'static str> = Some("wgpu_external_frame_ahardware_buffer");

    fn new(
        description: &HardwareBufferDesc,
        properties: &vulkan::BufferProperties,
    ) -> Result<Self, HardwareBufferImportError> {
        let format = match properties.format {
            vk::Format::UNDEFINED => {
                return Err(HardwareBufferImportError::ExternalFormat {
                    format: description.format,
                    external_format: properties.external_format,
                });
            }
            vk::Format::R8G8B8A8_UNORM => wgpu::TextureFormat::Rgba8Unorm,
            vk::Format::G8_B8R8_2PLANE_420_UNORM => wgpu::TextureFormat::NV12,
            other => {
                return Err(HardwareBufferImportError::UnsupportedFormat {
                    format: description.format,
                    vulkan_format: other.as_raw(),
                });
            }
        };
        let size = wgpu::Extent3d {
            width: description.width,
            height: description.height,
            depth_or_array_layers: 1,
        };
        let mip_level_count = if description
            .usage
            .contains(HardwareBufferUsage::GPU_MIPMAP_COMPLETE)
        {
            size.max_mips(wgpu::TextureDimension::D2)
        } else {
            1
        };
        let multi_planar = format.is_multi_planar_format();
        let (width_multiple, height_multiple) = format.size_multiple_requirement();
        assert!(
            size.width.is_multiple_of(width_multiple)
                && size.height.is_multiple_of(height_multiple)
                && (!multi_planar || mip_level_count == 1),
            "a {format:?} texture must be a single mip level whose extent is a multiple of \
             {width_multiple}x{height_multiple}, but the AHardwareBuffer is {}x{} with \
             {mip_level_count} levels",
            size.width,
            size.height,
        );
        // `wgpu` only samples multi-planar textures; it cannot copy their
        // planes.
        let usage = if multi_planar {
            wgpu::TextureUsages::TEXTURE_BINDING
        } else {
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC
        };
        Ok(Self {
            format,
            size,
            mip_level_count,
            usage,
        })
    }

    fn image_shape(&self, format: vk::Format) -> ImageShape {
        let mut usage = vk::ImageUsageFlags::SAMPLED;
        if self.usage.contains(wgpu::TextureUsages::COPY_SRC) {
            usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        ImageShape {
            format,
            width: self.size.width,
            height: self.size.height,
            mip_levels: self.mip_level_count,
            multi_planar: self.format.is_multi_planar_format(),
            usage,
        }
    }

    fn hal_descriptor(&self) -> wgpu::hal::TextureDescriptor<'static> {
        let mut usage = wgpu::TextureUses::RESOURCE;
        if self.usage.contains(wgpu::TextureUsages::COPY_SRC) {
            usage |= wgpu::TextureUses::COPY_SRC;
        }
        wgpu::hal::TextureDescriptor {
            label: Self::LABEL,
            size: self.size,
            mip_level_count: self.mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: Vec::new(),
        }
    }

    const fn descriptor(&self) -> wgpu::TextureDescriptor<'static> {
        wgpu::TextureDescriptor {
            label: Self::LABEL,
            size: self.size,
            mip_level_count: self.mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: self.usage,
            view_formats: &[],
        }
    }
}

/// Everything an import keeps alive for as long as `wgpu` uses its texture.
struct Retirement {
    imported: ImportedImage,
    buffer: BufferReference,
    lease: Option<Box<dyn HardwareBufferLease>>,
}

impl Retirement {
    /// Releases the import in dependency order: Vulkan's objects (and the
    /// buffer reference Vulkan's memory holds), then this side's buffer
    /// reference, then the producer's lease.
    fn retire(self) {
        let Self {
            imported,
            buffer,
            lease,
        } = self;
        drop(imported);
        drop(buffer);
        if let Some(lease) = lease {
            lease.release();
        }
    }
}
