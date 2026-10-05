# wgpu-external-frame

Import an externally produced GPU frame into [`wgpu`](https://wgpu.rs) as a
texture, without copying it through the CPU.

Browser engines, media decoders, screen capture, and compositors all publish
their output as a platform handle to memory the GPU already holds. This crate
turns each of those handles into a `wgpu::Texture` on a device you already own:

| Platform | Handle | Import path |
| --- | --- | --- |
| Linux | DMA-BUF file descriptor | Vulkan `VK_EXT_external_memory_dma_buf`, or EGL `EGL_LINUX_DMA_BUF_EXT` + `glEGLImageTargetTexture2DOES` |
| Android | `AHardwareBuffer` | Vulkan `VK_ANDROID_external_memory_android_hardware_buffer`; buffers of an external format are converted through `VK_KHR_push_descriptor` |
| macOS, iOS | `IOSurface` | `MTLDevice newTextureWithDescriptor:iosurface:plane:`, one texture per plane |
| Windows | Shared texture `HANDLE` | `ID3D12Device::OpenSharedHandle` |

Each platform is its own module with its own frame types — `DmaBufFrame`,
`HardwareBufferFrame`, `PackedIoSurfaceFrame` and `Ycbcr420IoSurfaceFrame`,
`SharedHandleFrame` — because the handles have nothing in common beyond the
goal. The Linux and Android sides additionally model the producer's *lease* on
the buffer (`DmaBufLease`, `HardwareBufferLease`), since those buffers usually
come from a pool the producer needs back, guarded by an explicit fence.

On Android an RGBA buffer imports as an `Rgba8Unorm` texture that aliases the
buffer itself, so nothing is copied. A 4:2:0 YCbCr buffer imports as two plane
views — luma `R8Unorm`, interleaved Cb/Cr `Rg8Unorm` — with the matrix and
range the Vulkan driver reports for it:

- when the driver maps the buffer to `G8_B8R8_2PLANE_420_UNORM`, the views are
  planes of one `NV12` texture that aliases the buffer;
- when the driver describes it only with an implementation-defined external
  format, as some drivers do for every YCbCr buffer, camera frames included,
  sampling it needs a `VkSamplerYcbcrConversion`, which `wgpu` cannot express.
  The import then converts it on the GPU, in a small raw Vulkan pass on the
  importer's queue, into two textures `wgpu` owns. No pixel passes through the
  CPU.

The device must be opened with the extensions and the YCbCr conversion feature
the import needs, which `wgpu` does not enable on its own;
`ahardware_buffer::request_device` does that. The conversion also needs
`VK_KHR_push_descriptor`, which `request_device` enables when the adapter
offers it. A device without it imports RGBA buffers and buffers the driver maps
to `NV12` as usual, and rejects a buffer that needs the conversion with
`HardwareBufferImportError::ConversionUnavailable`, which names the missing
extension. Building for Android compiles the
conversion's GLSL shaders with the NDK's `glslc`, found through
`ANDROID_NDK_HOME` or `ANDROID_NDK_ROOT`.

On Apple platforms, camera and video frames are biplanar 4:2:0 YCbCr surfaces,
and each of their planes imports as its own texture at that plane's extent:

| `IOSurface` pixel format | Frame type | Texture per plane | Device feature |
| --- | --- | --- | --- |
| packed BGRA or RGBA | `PackedIoSurfaceFrame` | `Bgra8Unorm` or `Rgba8Unorm` | — |
| `420v`, `420f` (8-bit 4:2:0) | `Ycbcr420IoSurfaceFrame` | luma `R8Unorm`, chroma `Rg8Unorm` | — |
| `x420`, `xf20` (10-bit 4:2:0) | `Ycbcr420IoSurfaceFrame` | luma `R16Unorm`, chroma `Rg16Unorm` | `TEXTURE_FORMAT_16BIT_NORM` |

Every import reaches through `wgpu` to its `hal` layer, so a device must be on
the backend its platform's import requires. Each entry point says which, and
asserts it.

## Status

This exists because `wgpu` has no portable external-memory import API yet. When
the work in the lineage of [wgpu#2320](https://github.com/gfx-rs/wgpu/issues/2320)
lands, most of what is here becomes a thin adapter over it, and the `hal`
reach-through goes away.
