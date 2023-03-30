//! Perturbed-Chain Statistical Associating Fluid Theory (PC-SAFT)
//!
//! [Gross et al. (2001)](https://doi.org/10.1021/ie0003887)
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "dft")]
mod dft;
mod eos;
pub mod parameters;

#[cfg(feature = "dft")]
pub use dft::{PcSaftFunctional, AttractiveFunctional, ChainFunctional, PSI_DFT, pure_saft_functional};
pub use eos::{DQVariants, PcSaft, PcSaftOptions, omega11, omega22};
pub use parameters::{PcSaftBinaryRecord, PcSaftParameters, PcSaftRecord};

#[cfg(feature = "python")]
pub mod python;
