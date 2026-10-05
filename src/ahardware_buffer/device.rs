use std::ffi::CStr;

use ash::vk;

/// The Vulkan device extensions an `AHardwareBuffer` import needs.
///
/// `wgpu` enables none of them on its own, so a device that imports hardware
/// buffers must be opened with them: through [`request_device`], or by adding
/// them in the callback of `wgpu::hal::vulkan::Adapter::open_with_callback`
/// for a renderer that opens its device itself.
///
/// - `VK_ANDROID_external_memory_android_hardware_buffer` imports the buffer's
///   memory and reports its format.
/// - `VK_EXT_queue_family_foreign` names the producer's side of the ownership
///   transfer every import performs.
/// - `VK_KHR_external_semaphore_fd` turns the producer's acquire fence into a
///   semaphore the GPU waits on.
///
/// The remaining dependencies of the hardware-buffer extension are core in
/// Vulkan 1.1, which [`request_device`] requires.
pub const DEVICE_EXTENSIONS: [&CStr; 3] = [
    ash::android::external_memory_android_hardware_buffer::NAME,
    ash::ext::queue_family_foreign::NAME,
    ash::khr::external_semaphore_fd::NAME,
];

/// Why [`request_device`] could not open a device.
#[derive(Debug, thiserror::Error)]
pub enum DeviceRequestError {
    /// The adapter is not a Vulkan 1.1 device, which the hardware-buffer
    /// extension and its dependencies require.
    #[error(
        "AHardwareBuffer import needs Vulkan 1.1, but the adapter reports Vulkan {major}.{minor}"
    )]
    VulkanVersion {
        /// The adapter's major Vulkan version.
        major: u32,
        /// The adapter's minor Vulkan version.
        minor: u32,
    },
    /// The adapter does not offer one of [`DEVICE_EXTENSIONS`].
    #[error("the Vulkan adapter does not support the device extension {0:?}")]
    MissingExtension(&'static CStr),
    /// The descriptor asks for features the adapter does not have.
    #[error("the adapter does not support the requested features {0:?}")]
    UnsupportedFeatures(wgpu::Features),
    /// The descriptor asks for experimental features without opting in to them.
    #[error(
        "experimental features {0:?} were requested, but experimental features are not enabled"
    )]
    ExperimentalFeaturesNotEnabled(wgpu::Features),
    /// The descriptor asks for a limit beyond what the adapter allows.
    #[error("the requested limit {name} = {requested} exceeds the adapter's {allowed}")]
    LimitExceeded {
        /// The limit's field name in `wgpu::Limits`.
        name: &'static str,
        /// The value the descriptor asked for.
        requested: u64,
        /// The adapter's bound for it.
        allowed: u64,
    },
    /// Vulkan failed to create the device.
    #[error("Vulkan failed to create the device: {0}")]
    Open(#[source] wgpu::hal::DeviceError),
    /// `wgpu` failed to adopt the device Vulkan created.
    #[error(transparent)]
    Create(#[from] wgpu::RequestDeviceError),
}

/// Opens a device on `adapter` that can import `AHardwareBuffer`s, enabling
/// [`DEVICE_EXTENSIONS`] alongside whatever `descriptor` asks for.
///
/// `wgpu` has no portable way to enable extra device extensions, so this opens
/// the device through `wgpu-hal`'s creation callback and hands it to `wgpu`.
/// That path skips the validation `wgpu::Adapter::request_device` performs, so
/// the features and limits of `descriptor` are checked against the adapter
/// here, with the same rules.
///
/// Request `wgpu::Features::TEXTURE_FORMAT_NV12` in `descriptor` to import
/// multi-planar YCbCr buffers.
///
/// # Errors
///
/// Returns an error when the adapter cannot provide Vulkan 1.1, one of the
/// extensions, or the requested features and limits, or when device creation
/// fails.
///
/// # Panics
///
/// Panics unless `adapter` is a Vulkan adapter.
pub fn request_device(
    adapter: &wgpu::Adapter,
    descriptor: &wgpu::DeviceDescriptor<'_>,
) -> Result<(wgpu::Device, wgpu::Queue), DeviceRequestError> {
    check_descriptor(adapter, descriptor)?;
    // SAFETY: `Adapter::as_hal` requires naming the adapter's real backend; a
    // non-Vulkan adapter yields `None` and panics here instead of being
    // reinterpreted. The guard is only used to open a device, which leaves the
    // adapter itself untouched.
    let hal_adapter = unsafe {
        adapter
            .as_hal::<wgpu::hal::api::Vulkan>()
            .expect("AHardwareBuffer import requires a Vulkan adapter")
    };
    let capabilities = hal_adapter.physical_device_capabilities();
    let api_version = capabilities.properties().api_version;
    if api_version < vk::API_VERSION_1_1 {
        return Err(DeviceRequestError::VulkanVersion {
            major: vk::api_version_major(api_version),
            minor: vk::api_version_minor(api_version),
        });
    }
    if let Some(missing) = DEVICE_EXTENSIONS
        .into_iter()
        .find(|extension| !capabilities.supports_extension(extension))
    {
        return Err(DeviceRequestError::MissingExtension(missing));
    }
    // SAFETY: `open_with_callback` has `Adapter::open`'s contract — features
    // and limits the adapter supports — plus the callback's: it may only add
    // what the device supports and must not remove anything.
    // `check_descriptor` verified the features and limits against this
    // adapter, and the callback only appends extensions that were just checked
    // with `supports_extension`, skipping any `wgpu` already enabled. Their
    // dependencies are core in the Vulkan 1.1 the device was checked to have.
    let open_device = unsafe {
        hal_adapter.open_with_callback(
            descriptor.required_features,
            &descriptor.required_limits,
            &descriptor.memory_hints,
            Some(Box::new(|arguments| {
                for extension in DEVICE_EXTENSIONS {
                    if !arguments.extensions.contains(&extension) {
                        arguments.extensions.push(extension);
                    }
                }
            })),
        )
    }
    .map_err(DeviceRequestError::Open)?;
    drop(hal_adapter);
    // SAFETY: `create_device_from_hal` requires a device opened from this
    // adapter with the features and limits `descriptor` describes; it was
    // opened just above from this adapter's own hal object, with exactly
    // `descriptor`'s features, limits, and memory hints.
    let device = unsafe { adapter.create_device_from_hal(open_device, descriptor) }?;
    Ok(device)
}

fn check_descriptor(
    adapter: &wgpu::Adapter,
    descriptor: &wgpu::DeviceDescriptor<'_>,
) -> Result<(), DeviceRequestError> {
    let unsupported = descriptor.required_features - adapter.features();
    if !unsupported.is_empty() {
        return Err(DeviceRequestError::UnsupportedFeatures(unsupported));
    }
    let experimental = descriptor
        .required_features
        .intersection(wgpu::Features::all_experimental_mask());
    if !experimental.is_empty() && !descriptor.experimental_features.is_enabled() {
        return Err(DeviceRequestError::ExperimentalFeaturesNotEnabled(
            experimental,
        ));
    }
    let mut exceeded = None;
    descriptor.required_limits.check_limits_with_fail_fn(
        &adapter.limits(),
        true,
        |name, requested, allowed| {
            exceeded.get_or_insert(DeviceRequestError::LimitExceeded {
                name,
                requested,
                allowed,
            });
        },
    );
    exceeded.map_or(Ok(()), Err)
}
