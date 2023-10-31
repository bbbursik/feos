use std::sync::Arc;

// use crate::hard_sphere::{FMTContribution, FMTFunctional, FMTProperties};
use crate::profile::DFTProfile;
use crate::{
    Convolver, ConvolverFFT, FunctionalContribution, HelmholtzEnergyFunctional, WeightFunction,
    WeightFunctionInfo, WeightFunctionShape,
};
use feos_core::si::SIUnit;
use feos_core::{Contributions, EosResult};
use ndarray::{Array, Array1, Array2, Axis, Dimension, Ix1, Ix2, RemoveAxis};
// use ndarray_npy::write_npy;

use num_dual::{Dual64, DualVec};
use feos_core::si::*;
// use feos_core::quantity::{QuantityArray, QuantityScalar};

/// entropy scaling trait for functional contributions --> provide a different set of weight functions
pub trait EntropyScalingFunctionalContribution: FunctionalContribution {
    fn weight_functions_entropy(&self, temperature: f64) -> WeightFunctionInfo<f64>;
}

// Trait EntropyScalingFunctional for the viscosity calculation
pub trait EntropyScalingFunctional: HelmholtzEnergyFunctional
// where
//     D: Dimension,
//     D::Larger: Dimension<Smaller = D>,
{
    fn entropy_scaling_contributions(&self) -> &[Box<dyn EntropyScalingFunctionalContribution>];

    /// Viscosity referaence for entropy scaling for the shear viscosity.
    fn viscosity_reference<D>(
        &self,
        density: &Density<Array<f64, D::Larger>>,
        temperature: Temperature,
    ) -> EosResult<Viscosity<D>>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>;

    // /// Correlation function for entropy scaling of the shear viscosity.
    // fn viscosity_correlation<D>(
    //     &self,
    //     s_res: &Array<f64, D>,
    //     density: &QuantityArray<U, D::Larger>,
    // ) -> EosResult<Array<f64, D>>
    // where
    //     D: Dimension,
    //     D::Larger: Dimension<Smaller = D>;

    // // add diffusion coefficient etc here
}

/// implement Entropy SCaling for viscosity profiles in 1D (only for functionals which implement the trait EntropyScalingFunctional)
impl<F> DFTProfile<Ix1, F>
where
    F: HelmholtzEnergyFunctional + EntropyScalingFunctional,
{
    // getter function for viscosity reference 
    pub fn viscosity_reference_1d(&self) -> EosResult<Viscosity<Ix1>> {
        self
        .dft
        .viscosity_reference::<Ix1>(&self.density, self.temperature)
    }

        // provide the weighted densities for entropy scaling
    pub fn weighted_densities_entropy(&self) -> EosResult<Vec<Array<f64, Ix2>>> {
        let temperature_red = self
            .temperature
            .to_reduced();

        let functional_contributions_entropy = self
        .dft
        .entropy_scaling_contributions();

        let weight_functions_entropy: Vec<WeightFunctionInfo<f64>> =
            functional_contributions_entropy
                .iter()
                .map(|c| c.weight_functions_entropy(temperature_red))
                .collect();
        let convolver_entropy: Arc<dyn Convolver<f64, Ix1>> =
            ConvolverFFT::plan(&self.grid, &weight_functions_entropy, None);

        let density_red = self.density.to_reduced();

        Ok(convolver_entropy.weighted_densities(&density_red))
    }
}