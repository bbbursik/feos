#![warn(clippy::all)]
#![allow(clippy::suspicious_operation_groupings)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::new_ret_no_self)]

pub mod adsorption;
pub mod convolver;
pub mod functional;
mod functional_contribution;
mod geometry;
mod ideal_chain_contribution;
pub mod interface;
mod pdgt;
pub mod profile;
pub mod solvation;
mod solver;
pub mod weight_functions;

pub use convolver::{Convolver, ConvolverFFT};
pub use functional::{HelmholtzEnergyFunctional, MoleculeShape, DFT};
pub use functional_contribution::{FunctionalContribution, FunctionalContributionDual};
pub use geometry::{Axis, Geometry, Grid};
pub use profile::{DFTProfile, DFTSpecification, DFTSpecifications};
pub use solver::{DFTSolver, DFTSolverLog};
pub use weight_functions::{WeightFunction, WeightFunctionInfo, WeightFunctionShape};

#[cfg(feature = "python")]
pub mod python;
