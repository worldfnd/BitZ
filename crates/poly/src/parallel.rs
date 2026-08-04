// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

/// Target number of `T` values in one cache-sized unit of work.
///
/// Adapted from WHIR's `workload_size`:
/// <https://github.com/worldfnd/whir/blob/e0aec15225fd5e63594bdc49566e080a6cab2f24/src/utils.rs#L27-L53>.
pub const fn workload_size<T: Sized>() -> usize {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    const CACHE_SIZE: usize = 1 << 17; // 128 KiB for Apple Silicon.

    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "ios", target_os = "android", target_os = "linux")
    ))]
    const CACHE_SIZE: usize = 1 << 16; // 64 KiB for mobile/server ARM.

    #[cfg(target_arch = "x86_64")]
    const CACHE_SIZE: usize = 1 << 15; // 32 KiB for x86-64.

    #[cfg(not(any(
        all(target_arch = "aarch64", target_os = "macos"),
        all(
            target_arch = "aarch64",
            any(target_os = "ios", target_os = "android", target_os = "linux")
        ),
        target_arch = "x86_64"
    )))]
    const CACHE_SIZE: usize = 1 << 15; // 32 KiB default.

    CACHE_SIZE / size_of::<T>()
}
