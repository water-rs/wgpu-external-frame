//! The GPU pass that converts an external-format 4:2:0 YCbCr buffer into a
//! luma and a chroma texture `wgpu` owns.
//!
//! An external-format image can only be read through a sampler carrying a
//! `VkSamplerYcbcrConversion`, bound as the immutable sampler of a combined
//! image sampler. `wgpu` cannot express either, so this pass is raw Vulkan:
//! two render passes that sample the buffer with the `RGB_IDENTITY` model —
//! the stored codes, unconverted — and write luma into an `R8Unorm` texture
//! and the Cb/Cr pairs into an `Rg8Unorm` texture at half the extent.
//!
//! The objects that depend only on the device ([`Shared`]) and those that
//! depend on the buffer's conversion parameters ([`ConversionPipeline`]) are
//! created once and cached; each frame creates only the views and
//! framebuffers that name its own images ([`ConversionFrame`]), and pushes the
//! descriptor that binds its source into the command buffer that converts it.

use core::ops::Deref;
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use ash::vk;
use wgpu::hal::Device as _;

use super::frame::FrameParts;
use super::vulkan::{self, BufferProperties, ImageFormat, ImageShape};
use super::{HardwareBufferUsage, Retirement, Ycbcr420Planes};
use crate::YcbcrEncoding;

/// The luma plane's texture format, as `wgpu` and Vulkan name it.
pub(super) const LUMA_FORMAT: (wgpu::TextureFormat, vk::Format) =
    (wgpu::TextureFormat::R8Unorm, vk::Format::R8_UNORM);

/// The chroma plane's texture format, as `wgpu` and Vulkan name it.
pub(super) const CHROMA_FORMAT: (wgpu::TextureFormat, vk::Format) =
    (wgpu::TextureFormat::Rg8Unorm, vk::Format::R8G8_UNORM);

const VERTEX_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ycbcr_planes.vert.spv"));
const FRAGMENT_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ycbcr_planes.frag.spv"));

/// Imports the external-format buffer of `parts` by converting it into a
/// luma and a chroma texture, and submits the conversion.
///
/// The conversion is submitted before this returns, so every later submission
/// that uses the planes runs after it. Once it has completed, the buffer and
/// every object the conversion created for this frame are released, and the
/// lease with them; the planes are ordinary `wgpu` textures from then on.
///
/// `hal_device` is dropped before the planes are handed to `wgpu`.
///
/// # Panics
///
/// Panics when the buffer is mipmapped or its extent is odd, which a 4:2:0
/// plane layout cannot represent, or when Vulkan fails to import or convert
/// it.
pub(super) fn import(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    converter: &mut Converter,
    hal_device: impl Deref<Target = wgpu::hal::vulkan::Device>,
    parts: FrameParts,
    properties: &BufferProperties,
    encoding: YcbcrEncoding,
) -> Ycbcr420Planes {
    let FrameParts {
        buffer,
        description,
        acquire_fence,
        mut lease,
    } = parts;
    assert!(
        description.width.is_multiple_of(2)
            && description.height.is_multiple_of(2)
            && !description
                .usage
                .contains(HardwareBufferUsage::GPU_MIPMAP_COMPLETE),
        "a 4:2:0 YCbCr AHardwareBuffer must be a single mip level of even extent, but this one \
         is {}x{} with usage {:?}",
        description.width,
        description.height,
        description.usage,
    );
    let extent = vk::Extent2D {
        width: description.width,
        height: description.height,
    };
    let pipeline = converter.pipeline(properties);
    let imported = vulkan::import_buffer(
        &hal_device,
        buffer.buffer(),
        properties,
        &ImageShape {
            format: ImageFormat::External(properties.external_format),
            width: extent.width,
            height: extent.height,
            mip_levels: 1,
            usage: vk::ImageUsageFlags::SAMPLED,
        },
        acquire_fence,
    );
    let luma = PlaneTexture::new(&hal_device, LUMA_FORMAT.0, extent.width, extent.height);
    let chroma = PlaneTexture::new(
        &hal_device,
        CHROMA_FORMAT.0,
        extent.width / 2,
        extent.height / 2,
    );
    let frame = ConversionFrame::new(
        pipeline,
        &ConversionTargets {
            source: imported.image(),
            luma: luma.image,
            chroma: chroma.image,
            extent,
        },
    );
    let source = imported.image();
    let acquire_semaphore = imported.acquire_semaphore();
    let raw_device = hal_device.raw_device().clone();
    let queue_family_index = hal_device.queue_family_index();
    drop(hal_device);
    let luma = luma.into_wgpu(device);
    let chroma = chroma.into_wgpu(device);
    if let Some(lease) = lease.as_mut() {
        lease.presented();
    }
    let conversion = super::record_raw(device, "wgpu_external_frame_ycbcr_conversion", |encoder| {
        // SAFETY: the encoder records into the submission below, which keeps
        // both planes alive until it completes, as `submit` describes; the
        // source image and `frame` are released only by the completion
        // callback registered below. The acquire barrier moves the source into
        // `SHADER_READ_ONLY_OPTIMAL` for the fragment shader before the
        // conversion reads it, and the planes were created just above, so no
        // submission has used them yet.
        unsafe {
            vulkan::record_acquire(&raw_device, encoder, source, queue_family_index);
            frame.record(encoder.raw_handle());
        }
    });
    let retirement = Retirement {
        imported,
        buffer,
        lease,
    };
    conversion.on_submitted_work_done(move || {
        // The views and framebuffers name the source image, so they go first.
        drop(frame);
        retirement.retire();
    });
    super::submit(
        device,
        queue,
        conversion,
        &[&luma, &chroma],
        acquire_semaphore,
    );
    Ycbcr420Planes {
        luma: luma.create_view(&wgpu::TextureViewDescriptor::default()),
        chroma: chroma.create_view(&wgpu::TextureViewDescriptor::default()),
        encoding,
    }
}

/// A plane texture created through `wgpu-hal`, before `wgpu` adopts it.
///
/// It is created through the hal layer rather than `wgpu::Device` because the
/// conversion writes it outside `wgpu`'s knowledge: `wgpu` adopts it as
/// already initialized, where a texture of its own would count as
/// uninitialized and be cleared before its first use, discarding the planes.
struct PlaneTexture {
    texture: wgpu::hal::vulkan::Texture,
    image: vk::Image,
    format: wgpu::TextureFormat,
    size: wgpu::Extent3d,
}

impl PlaneTexture {
    const LABEL: Option<&'static str> = Some("wgpu_external_frame_ycbcr_plane");
    /// Sampled and copied by the consumer, rendered to by the conversion.
    const USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
        .union(wgpu::TextureUsages::COPY_SRC)
        .union(wgpu::TextureUsages::RENDER_ATTACHMENT);

    fn new(
        hal_device: &wgpu::hal::vulkan::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let descriptor = wgpu::hal::TextureDescriptor {
            label: Self::LABEL,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUses::RESOURCE
                | wgpu::TextureUses::COPY_SRC
                | wgpu::TextureUses::COLOR_TARGET,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: Vec::new(),
        };
        // SAFETY: the descriptor names a single-level 2D texture of a format
        // every device can sample, copy and render to, at a nonzero extent
        // within the device's limits since it is at most the buffer's.
        let texture = unsafe { hal_device.create_texture(&descriptor) }.unwrap_or_else(|error| {
            panic!("failed to create a {format:?} YCbCr plane texture: {error}")
        });
        // SAFETY: the texture was just created by `wgpu-hal` and owns its
        // image, which stays alive at least as long as the texture.
        let image = unsafe { texture.raw_handle() };
        Self {
            texture,
            image,
            format,
            size,
        }
    }

    /// Hands the texture to `wgpu`, which owns it from then on.
    fn into_wgpu(self, device: &wgpu::Device) -> wgpu::Texture {
        // SAFETY: the hal texture was created on this device's hal device from
        // a descriptor that matches this one, usage for usage. It counts as
        // initialized because the conversion, submitted before any use `wgpu`
        // records, writes every texel; its render passes leave it in
        // `SHADER_READ_ONLY_OPTIMAL`, made visible to the shader stages, which
        // is what `wgpu` assumes of the declared `RESOURCE` state.
        unsafe {
            device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                self.texture,
                &wgpu::TextureDescriptor {
                    label: Self::LABEL,
                    size: self.size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.format,
                    usage: Self::USAGE,
                    view_formats: &[],
                },
                wgpu::TextureUses::RESOURCE,
            )
        }
    }
}

/// The parameters a `VkSamplerYcbcrConversion` is created from, and so the
/// key its cached pipeline is found by.
///
/// The model is always `RGB_IDENTITY`, which ignores the range, and an
/// external-format conversion ignores its component swizzle, so neither is
/// part of the key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ConversionKey {
    external_format: u64,
    x_chroma_offset: vk::ChromaLocation,
    y_chroma_offset: vk::ChromaLocation,
}

/// Converts external-format buffers on one device, caching a pipeline per
/// set of conversion parameters.
pub(super) struct Converter {
    shared: Arc<Shared>,
    pipelines: HashMap<ConversionKey, Arc<ConversionPipeline>>,
}

impl core::fmt::Debug for Converter {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Converter")
            .field("pipelines", &self.pipelines.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl Converter {
    /// Creates the device-wide objects of the conversion.
    ///
    /// `hal_device` must enable every one of [`super::DEVICE_EXTENSIONS`],
    /// which `HardwareBufferImporter::new` asserts.
    pub(super) fn new(hal_device: &wgpu::hal::vulkan::Device) -> Self {
        Self {
            shared: Arc::new(Shared::new(
                hal_device.shared_instance().raw_instance(),
                hal_device.raw_device(),
            )),
            pipelines: HashMap::new(),
        }
    }

    /// The pipeline that converts buffers with `properties`, created on first
    /// use.
    pub(super) fn pipeline(&mut self, properties: &BufferProperties) -> Arc<ConversionPipeline> {
        let key = ConversionKey {
            external_format: properties.external_format,
            x_chroma_offset: properties.x_chroma_offset,
            y_chroma_offset: properties.y_chroma_offset,
        };
        Arc::clone(
            self.pipelines.entry(key).or_insert_with(|| {
                Arc::new(ConversionPipeline::new(Arc::clone(&self.shared), key))
            }),
        )
    }
}

/// The conversion's objects that depend only on the device: the shaders, the
/// render passes that write each plane, and the `VK_KHR_push_descriptor`
/// entry points that bind the source.
struct Shared {
    device: ash::Device,
    push_descriptor: ash::khr::push_descriptor::Device,
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
    luma_pass: vk::RenderPass,
    chroma_pass: vk::RenderPass,
}

impl Shared {
    /// `device` must have been created from `instance` with
    /// `VK_KHR_push_descriptor` enabled.
    fn new(instance: &ash::Instance, device: &ash::Device) -> Self {
        let vertex = shader_module(device, VERTEX_SPIRV);
        let fragment = shader_module(device, FRAGMENT_SPIRV);
        Self {
            device: device.clone(),
            push_descriptor: ash::khr::push_descriptor::Device::new(instance, device),
            vertex,
            fragment,
            luma_pass: plane_render_pass(device, LUMA_FORMAT.1),
            chroma_pass: plane_render_pass(device, CHROMA_FORMAT.1),
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: these objects were created on `self.device` by `new`, are
        // owned solely by this value and destroyed once. Every pipeline and
        // framebuffer created from them holds an `Arc` on this value, so none
        // is still alive, and so no pending submission uses them.
        unsafe {
            self.device.destroy_render_pass(self.chroma_pass, None);
            self.device.destroy_render_pass(self.luma_pass, None);
            self.device.destroy_shader_module(self.fragment, None);
            self.device.destroy_shader_module(self.vertex, None);
        }
    }
}

fn shader_module(device: &ash::Device, spirv: &[u8]) -> vk::ShaderModule {
    let code = ash::util::read_spv(&mut Cursor::new(spirv))
        .expect("the build script emits whole SPIR-V words");
    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
    // SAFETY: `code` is SPIR-V that `glslc` compiled for Vulkan 1.1 when the
    // crate was built, and `create_info` borrows it for the call.
    unsafe { device.create_shader_module(&create_info, None) }.unwrap_or_else(|error| {
        panic!("failed to create the YCbCr conversion shader module: {error}")
    })
}

/// A render pass with one color attachment of `format` that the pass
/// overwrites entirely and leaves ready for `wgpu` to sample.
fn plane_render_pass(device: &ash::Device, format: vk::Format) -> vk::RenderPass {
    let attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        // Every texel is written, so the previous contents are not needed.
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        // The layout `wgpu` expects of a texture in the `RESOURCE` state,
        // which the plane textures are declared to start in.
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let color = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(core::slice::from_ref(&color));
    let dependencies = [
        // Orders the transition out of `UNDEFINED` and the writes after any
        // earlier use of the memory on this queue.
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
        // Makes the writes visible to the stages and access `wgpu` assumes
        // for a texture in the `RESOURCE` state. A later `wgpu` barrier out
        // of that state names the same stages as its source, so it chains
        // after this dependency whatever the texture is used for next.
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(
                vk::PipelineStageFlags::VERTEX_SHADER
                    | vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
            )
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ),
    ];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(core::slice::from_ref(&attachment))
        .subpasses(core::slice::from_ref(&subpass))
        .dependencies(&dependencies);
    // SAFETY: `create_info` and every array it points to are locals that
    // outlive the call. The one attachment is a color format every Vulkan
    // device can render to, referenced in `COLOR_ATTACHMENT_OPTIMAL`, and the
    // dependencies name only stages and accesses of the graphics pipeline.
    unsafe { device.create_render_pass(&create_info, None) }.unwrap_or_else(|error| {
        panic!("failed to create the YCbCr conversion render pass: {error}")
    })
}

/// The conversion's objects for one set of conversion parameters: the
/// `VkSamplerYcbcrConversion`, its immutable sampler, and the pipelines that
/// sample through it.
pub(super) struct ConversionPipeline {
    shared: Arc<Shared>,
    conversion: vk::SamplerYcbcrConversion,
    sampler: vk::Sampler,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    luma: vk::Pipeline,
    chroma: vk::Pipeline,
}

impl ConversionPipeline {
    fn new(shared: Arc<Shared>, key: ConversionKey) -> Self {
        let device = &shared.device;
        let mut external_format =
            vk::ExternalFormatANDROID::default().external_format(key.external_format);
        let conversion_info = vk::SamplerYcbcrConversionCreateInfo::default()
            .push_next(&mut external_format)
            .format(vk::Format::UNDEFINED)
            // Passes the stored codes through: luma in green, Cb in blue, Cr
            // in red. The consumer applies the matrix the import reports.
            .ycbcr_model(vk::SamplerYcbcrModelConversion::RGB_IDENTITY)
            .ycbcr_range(vk::SamplerYcbcrRange::ITU_FULL)
            .components(vk::ComponentMapping::default())
            .x_chroma_offset(key.x_chroma_offset)
            .y_chroma_offset(key.y_chroma_offset)
            .chroma_filter(vk::Filter::NEAREST)
            .force_explicit_reconstruction(false);
        // SAFETY: the device was opened with the `samplerYcbcrConversion`
        // feature (`request_device`, `DeviceRequirements`). The struct and its
        // chained external format are locals that outlive the call. An
        // external-format conversion has format `UNDEFINED`, and the chroma
        // offsets are the ones the driver suggested for this external format,
        // so its format features support them.
        let conversion = unsafe { device.create_sampler_ycbcr_conversion(&conversion_info, None) }
            .unwrap_or_else(|error| {
                panic!(
                    "failed to create the VkSamplerYcbcrConversion for external format {:#x}: \
                     {error}",
                    key.external_format
                )
            });
        let mut conversion_binding =
            vk::SamplerYcbcrConversionInfo::default().conversion(conversion);
        let sampler_info = vk::SamplerCreateInfo::default()
            .push_next(&mut conversion_binding)
            // A sampler with a YCbCr conversion must filter as the conversion's
            // chroma filter does, clamp to the edge, and use normalized
            // coordinates without anisotropy or comparison.
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .compare_enable(false)
            .unnormalized_coordinates(false);
        // SAFETY: the conversion was just created on `device`, and the sampler
        // parameters are the ones a sampler with a YCbCr conversion must have.
        let sampler =
            unsafe { device.create_sampler(&sampler_info, None) }.unwrap_or_else(|error| {
                panic!("failed to create the YCbCr conversion sampler: {error}")
            });
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .immutable_samplers(core::slice::from_ref(&sampler));
        // The source is bound with a push descriptor rather than a descriptor
        // set, because a set must come from a pool, and the specification
        // gives the number of pool descriptors a combined image sampler of an
        // Android external format consumes only through
        // `maxCombinedImageSamplerDescriptorCount` (the note under
        // `VkDescriptorPoolSize`), which only Vulkan 1.4 and
        // `VK_KHR_maintenance6` report. Push descriptors are stored by the
        // command buffer ("whose storage is internally managed by the command
        // buffer", `vkCmdPushDescriptorSet`), so no count is needed.
        //
        // A push-descriptor layout may carry an immutable YCbCr sampler:
        // - `VUID-VkDescriptorSetLayoutCreateInfo-flags-00280`, `-02208` and
        //   `-04591` exclude only dynamic buffers, inline uniform blocks and
        //   mutable descriptors from such a layout, and
        //   `VUID-VkDescriptorSetLayoutBinding-descriptorType-12200` only asks
        //   that every immutable sampler of the binding enable a conversion,
        //   as this one does;
        // - `vkCmdPushDescriptorSet` says of a `COMBINED_IMAGE_SAMPLER` write
        //   that "the sampler member of the pImageInfo parameter is ignored
        //   and the immutable sampler is taken from the push descriptor set
        //   layout in the pipeline layout";
        // - `VUID-VkDescriptorSetLayoutCreateInfo-flags-00281` bounds the
        //   layout's elements by `maxPushDescriptors`, whose required minimum
        //   is 32; this layout has one.
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(core::slice::from_ref(&binding));
        // SAFETY: a YCbCr conversion sampler must be bound as an immutable
        // sampler of a combined image sampler, which this one binding is; the
        // sampler was created on `device` just above. The push-descriptor flag
        // needs `VK_KHR_push_descriptor`, which the device enables, and is
        // valid for this binding as the comment above sets out.
        let set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None) }
            .unwrap_or_else(|error| {
                panic!("failed to create the YCbCr conversion descriptor set layout: {error}")
            });
        let layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(core::slice::from_ref(&set_layout));
        // SAFETY: the one set layout was created on `device` just above.
        let layout =
            unsafe { device.create_pipeline_layout(&layout_info, None) }.unwrap_or_else(|error| {
                panic!("failed to create the YCbCr conversion pipeline layout: {error}")
            });
        let [luma, chroma] = plane_pipelines(&shared, layout);
        Self {
            shared,
            conversion,
            sampler,
            set_layout,
            layout,
            luma,
            chroma,
        }
    }
}

impl Drop for ConversionPipeline {
    fn drop(&mut self) {
        let device = &self.shared.device;
        // SAFETY: these objects were created on `device` by `new`, are owned
        // solely by this value and destroyed once, dependents first. Every
        // frame that records them holds an `Arc` on this value until the
        // submission it was recorded into has completed, so no pending
        // submission uses them.
        unsafe {
            device.destroy_pipeline(self.chroma, None);
            device.destroy_pipeline(self.luma, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_sampler_ycbcr_conversion(self.conversion, None);
        }
    }
}

/// The luma and chroma pipelines: the same shaders, specialized by the plane
/// they write, each compatible with that plane's render pass.
fn plane_pipelines(shared: &Shared, layout: vk::PipelineLayout) -> [vk::Pipeline; 2] {
    let specialization_entry = vk::SpecializationMapEntry::default()
        .constant_id(0)
        .offset(0)
        .size(size_of::<vk::Bool32>());
    let luma_constant = vk::FALSE.to_ne_bytes();
    let chroma_constant = vk::TRUE.to_ne_bytes();
    let luma_specialization = vk::SpecializationInfo::default()
        .map_entries(core::slice::from_ref(&specialization_entry))
        .data(&luma_constant);
    let chroma_specialization = vk::SpecializationInfo::default()
        .map_entries(core::slice::from_ref(&specialization_entry))
        .data(&chroma_constant);
    let stages = |specialization| {
        [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shared.vertex)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shared.fragment)
                .name(c"main")
                .specialization_info(specialization),
        ]
    };
    let luma_stages = stages(&luma_specialization);
    let chroma_stages = stages(&chroma_specialization);
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    // The viewport and scissor are dynamic, so one pipeline serves every
    // frame size.
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let luma_blend = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::R);
    let chroma_blend = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G);
    let luma_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(core::slice::from_ref(&luma_blend));
    let chroma_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(core::slice::from_ref(&chroma_blend));
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let common = vk::GraphicsPipelineCreateInfo::default()
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .dynamic_state(&dynamic)
        .layout(layout)
        .subpass(0);
    let create_infos = [
        common
            .stages(&luma_stages)
            .color_blend_state(&luma_blend_state)
            .render_pass(shared.luma_pass),
        common
            .stages(&chroma_stages)
            .color_blend_state(&chroma_blend_state)
            .render_pass(shared.chroma_pass),
    ];
    // SAFETY: every create info and the state it points to are locals that
    // outlive the call. The shader modules, render passes and layout were
    // created on `shared.device`; each pipeline targets subpass 0 of a render
    // pass with one color attachment, matching the fragment shader's one
    // output, and the layout's one set matches the shader's one binding.
    let pipelines = unsafe {
        shared
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &create_infos, None)
    }
    .unwrap_or_else(|(_, error)| {
        panic!("failed to create the YCbCr conversion pipelines: {error}")
    });
    pipelines
        .try_into()
        .expect("two create infos yield two pipelines")
}

/// The images one conversion reads and writes.
pub(super) struct ConversionTargets {
    /// The imported external-format image.
    pub(super) source: vk::Image,
    /// The `R8Unorm` image, at the frame's extent.
    pub(super) luma: vk::Image,
    /// The `Rg8Unorm` image, at half the frame's extent.
    pub(super) chroma: vk::Image,
    /// The frame's extent; both dimensions are even.
    pub(super) extent: vk::Extent2D,
}

/// One frame's conversion objects: the views and framebuffers that name its
/// images.
///
/// They are used only by the submission that records the conversion, and are
/// destroyed once it has completed.
pub(super) struct ConversionFrame {
    pipeline: Arc<ConversionPipeline>,
    extent: vk::Extent2D,
    source_view: vk::ImageView,
    luma_view: vk::ImageView,
    chroma_view: vk::ImageView,
    luma_framebuffer: vk::Framebuffer,
    chroma_framebuffer: vk::Framebuffer,
}

impl ConversionFrame {
    pub(super) fn new(pipeline: Arc<ConversionPipeline>, targets: &ConversionTargets) -> Self {
        let device = &pipeline.shared.device;
        let mut conversion_binding =
            vk::SamplerYcbcrConversionInfo::default().conversion(pipeline.conversion);
        let source_view_info = vk::ImageViewCreateInfo::default()
            .push_next(&mut conversion_binding)
            .image(targets.source)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::UNDEFINED)
            .components(vk::ComponentMapping::default())
            .subresource_range(color_subresource());
        // SAFETY: `targets.source` is a live external-format image created on
        // this device. A view of it must have format `UNDEFINED`, identity
        // swizzles, and the conversion created for the same external format,
        // which `pipeline` was looked up by.
        let source_view = unsafe { device.create_image_view(&source_view_info, None) }
            .unwrap_or_else(|error| {
                panic!("failed to create the view of an external-format AHardwareBuffer: {error}")
            });
        let mut frame = Self {
            extent: targets.extent,
            source_view,
            luma_view: vk::ImageView::null(),
            chroma_view: vk::ImageView::null(),
            luma_framebuffer: vk::Framebuffer::null(),
            chroma_framebuffer: vk::Framebuffer::null(),
            pipeline,
        };
        let chroma_extent = vk::Extent2D {
            width: targets.extent.width / 2,
            height: targets.extent.height / 2,
        };
        frame.luma_view = frame.plane_view(targets.luma, LUMA_FORMAT.1);
        frame.chroma_view = frame.plane_view(targets.chroma, CHROMA_FORMAT.1);
        frame.luma_framebuffer = frame.framebuffer(
            frame.pipeline.shared.luma_pass,
            frame.luma_view,
            targets.extent,
        );
        frame.chroma_framebuffer = frame.framebuffer(
            frame.pipeline.shared.chroma_pass,
            frame.chroma_view,
            chroma_extent,
        );
        frame
    }

    fn device(&self) -> &ash::Device {
        &self.pipeline.shared.device
    }

    fn plane_view(&self, image: vk::Image, format: vk::Format) -> vk::ImageView {
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping::default())
            .subresource_range(color_subresource());
        // SAFETY: `image` is a live single-level, single-layer image of
        // `format`, created by `wgpu-hal` with color-attachment usage.
        unsafe { self.device().create_image_view(&create_info, None) }.unwrap_or_else(|error| {
            panic!("failed to create the view of a YCbCr plane texture: {error}")
        })
    }

    fn framebuffer(
        &self,
        render_pass: vk::RenderPass,
        view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> vk::Framebuffer {
        let create_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(core::slice::from_ref(&view))
            .width(extent.width)
            .height(extent.height)
            .layers(1);
        // SAFETY: `view` is a view of an image of exactly `extent`, in the
        // format of `render_pass`'s one attachment.
        unsafe { self.device().create_framebuffer(&create_info, None) }.unwrap_or_else(|error| {
            panic!("failed to create the framebuffer of a YCbCr plane: {error}")
        })
    }

    /// Records the conversion: luma, then chroma.
    ///
    /// # Safety
    ///
    /// `command_buffer` must be recording, outside a render pass, into the
    /// submission that keeps this frame alive until it completes, after the
    /// barrier that acquires the source image into
    /// `SHADER_READ_ONLY_OPTIMAL` for the fragment shader. The plane images
    /// must not be in use by any pending submission.
    pub(super) unsafe fn record(&self, command_buffer: vk::CommandBuffer) {
        let chroma_extent = vk::Extent2D {
            width: self.extent.width / 2,
            height: self.extent.height / 2,
        };
        // The sampler is ignored: the binding's immutable YCbCr sampler is
        // used instead.
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(self.source_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let write = vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(core::slice::from_ref(&image_info));
        // SAFETY: the caller guarantees a recording command buffer. Set 0 is
        // the pipeline layout's only set, created with the push-descriptor
        // flag, and the write fills its one combined image sampler with a
        // view of the source created with the same conversion as the
        // binding's immutable sampler, in the layout the acquire barrier moves
        // the image into before it is read. Both plane pipelines share this
        // layout, so binding either keeps the pushed descriptor, which every
        // draw below reads. The plane recordings rest on the caller's
        // contract, with each plane's framebuffer, pipeline and extent
        // belonging together.
        unsafe {
            self.pipeline
                .shared
                .push_descriptor
                .cmd_push_descriptor_set(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.layout,
                    0,
                    core::slice::from_ref(&write),
                );
            self.record_plane(
                command_buffer,
                self.luma_framebuffer,
                self.pipeline.shared.luma_pass,
                self.pipeline.luma,
                self.extent,
            );
            self.record_plane(
                command_buffer,
                self.chroma_framebuffer,
                self.pipeline.shared.chroma_pass,
                self.pipeline.chroma,
                chroma_extent,
            );
        }
    }

    /// # Safety
    ///
    /// As [`Self::record`], after the source's descriptor has been pushed,
    /// and `framebuffer`, `render_pass` and `pipeline` must be one plane's, at
    /// that plane's `extent`.
    unsafe fn record_plane(
        &self,
        command_buffer: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
        render_pass: vk::RenderPass,
        pipeline: vk::Pipeline,
        extent: vk::Extent2D,
    ) {
        let device = self.device();
        let area = vk::Rect2D::default().extent(extent);
        let begin = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(area);
        #[expect(
            clippy::cast_precision_loss,
            reason = "a Vulkan image extent is far below the 2^24 an f32 holds exactly"
        )]
        let viewport = vk::Viewport::default()
            .width(extent.width as f32)
            .height(extent.height as f32)
            .max_depth(1.0);
        // SAFETY: the caller guarantees a recording command buffer outside a
        // render pass, and that the framebuffer, render pass and pipeline are
        // one plane's. The render area is the framebuffer's whole extent, the
        // load op needs no clear values, the dynamic viewport and scissor are
        // set before the draw, and the pipeline's one descriptor has been
        // pushed. The draw of three vertices needs no vertex buffer.
        unsafe {
            device.cmd_begin_render_pass(command_buffer, &begin, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[area]);
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
            device.cmd_end_render_pass(command_buffer);
        }
    }
}

impl Drop for ConversionFrame {
    fn drop(&mut self) {
        let device = self.device();
        // SAFETY: every handle was created on this device by `new` — or is
        // still null if `new` panicked part-way, which the destroy calls
        // accept — is owned solely by this value and destroyed once,
        // dependents first. A frame is dropped only once the submission that
        // recorded it has completed, so the GPU no longer uses any of them.
        unsafe {
            device.destroy_framebuffer(self.chroma_framebuffer, None);
            device.destroy_framebuffer(self.luma_framebuffer, None);
            device.destroy_image_view(self.chroma_view, None);
            device.destroy_image_view(self.luma_view, None);
            device.destroy_image_view(self.source_view, None);
        }
    }
}

const fn color_subresource() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}
