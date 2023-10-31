use std::rc::Rc;

use crate::fundamental_measure_theory::{FMTContribution, FMTFunctional, FMTProperties};
use crate::profile::DFTProfile;
use crate::{
    Convolver, ConvolverFFT, FunctionalContribution, HelmholtzEnergyFunctional, WeightFunction,
    WeightFunctionInfo, WeightFunctionShape,
};
use feos_core::{Contributions, EosResult, EosUnit};
use ndarray::{Array, Array1, Array2, Axis, Dimension, Ix1, Ix2, RemoveAxis};
use ndarray_npy::write_npy;

use num_dual::{Dual64, DualVec};
use quantity::si::*;
use quantity::{QuantityArray, QuantityScalar};

/// entropy scaling trait for functional contributions --> provide a different set of weight functions
pub trait EntropyScalingFunctionalContribution: FunctionalContribution {
    fn weight_functions_entropy(&self, temperature: f64) -> WeightFunctionInfo<f64>;
}

/// implement weight functions for entropy scaling for FMTContribution
impl<P: FMTProperties> EntropyScalingFunctionalContribution for FMTContribution<P> {
    fn weight_functions_entropy(&self, temperature: f64) -> WeightFunctionInfo<f64> {
        let r = self.properties.hs_diameter(temperature) * 0.5;

        // compare to the actual weight functions for FMT
        WeightFunctionInfo::new(self.properties.component_index(), false)
            .add(
                WeightFunction::new_scaled(r.clone(), WeightFunctionShape::Theta),
                true,
            )
            .add(
                WeightFunction::new_scaled(r.clone(), WeightFunctionShape::Delta),
                true,
            )
            .add(
                WeightFunction::new_scaled(r.clone(), WeightFunctionShape::DeltaVec),
                true,
            )
    }
}

// Trait EntropyScalingFunctional for the viscosity calculation
pub trait EntropyScalingFunctional<U: EosUnit>: HelmholtzEnergyFunctional
// where
//     D: Dimension,
//     D::Larger: Dimension<Smaller = D>,
{
    fn entropy_scaling_contributions(&self) -> &[Box<dyn EntropyScalingFunctionalContribution>];

    /// Viscosity referaence for entropy scaling for the shear viscosity.
    fn viscosity_reference<D>(
        &self,
        density: &QuantityArray<U, D::Larger>,
        temperature: QuantityScalar<U>,
    ) -> EosResult<QuantityArray<U, D>>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>;

    /// Correlation function for entropy scaling of the shear viscosity.
    fn viscosity_correlation<D>(
        &self,
        s_res: &Array<f64, D>,
        density: &QuantityArray<U, D::Larger>,
    ) -> EosResult<Array<f64, D>>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>;

    // add diffusion coefficient etc here
}

/// implement Entropy SCaling for viscosity profiles in 1D (only for functionals which implement the trait EntropyScalingFunctional)
impl<F> DFTProfile<SIUnit, Ix1, F>
where
    F: HelmholtzEnergyFunctional + EntropyScalingFunctional<SIUnit>,
{
    // getter function for viscosity reference 
    pub fn viscosity_reference_1d(&self) -> EosResult<SIArray<Ix1>> {
        self
        .dft
        .viscosity_reference::<Ix1>(&self.density, self.temperature)
    }

    
}