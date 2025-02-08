//! This module holds entities that allow observing and measuring driver's and cluster's behaviour.
//! This includes:
//! - driver-side tracing,
//! - cluster-side tracing,
//! - request execution history,
//! - lock-free histogram,
//! - driver metrics.

pub(crate) mod driver_tracing;
pub mod history;
pub(crate) mod lock_free_histogram;
pub mod metrics;
pub mod tracing;
