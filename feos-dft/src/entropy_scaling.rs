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
    /// this function actually calcuates the viscosity profile and is called from python 
    pub fn viscosity_profile_1d(&self) -> EosResult<SIArray<Ix1>> {
        let temperature_red = self
            .temperature
            .to_reduced(SIUnit::reference_temperature())?;
        
            let density_red = self.density.to_reduced(SIUnit::reference_density())?;

        // initialize the convolver
        let weight_functions_dual: Vec<WeightFunctionInfo<Dual64>> = self
            .dft
            .functional
            .contributions()
            .iter()
            .map(|c| c.weight_functions(Dual64::from(temperature_red).derive()))
            .collect();

        let convolver_dual: Rc<dyn Convolver<DualVec<f64, f64, 1>, Ix1>> =
            ConvolverFFT::plan(&self.grid, &weight_functions_dual, None);

        //// Initialize entropy convolver
        // get the entropy scaling contributions which are defined in the implementation of EnropyScalingFunctional
        let functional_contributions_entropy = self
        .dft
        .functional
        .entropy_scaling_contributions();
        
        // get the entropy scaling weight functions (defined in the contributions)
        let weight_functions_entropy: Vec<WeightFunctionInfo<f64>> =
            functional_contributions_entropy
                .iter()
                .map(|c| c.weight_functions_entropy(temperature_red))
                .collect();
        
        let convolver_entropy: Rc<dyn Convolver<f64, Ix1>> =
            ConvolverFFT::plan(&self.grid, &weight_functions_entropy, None);

        // Code (originally placed in entropy_density_contributions)

        // Weighted densities
        let weighted_densities_entropy: Vec<Array2<f64>> =
            convolver_entropy.weighted_densities(&density_red);

        // for (i, wd_e) in weighted_densities_entropy.iter().enumerate() {
        //     let filename0 = i.to_string().to_owned();
        //     write_npy(filename0 + "_wd_entropy.npy", wd_e).unwrap();
        // }

        // Molar entropy calculation for each contribution (entropy density divided by weighted density)
        // the unit conversion is necessary to obtain the molar (and not per molecule) entropy
        let entropy_molar_contributions = self
            .dft
            .entropy_density_contributions::<Ix1>( // calculates the entropy density
                temperature_red,
                &density_red,
                &convolver_dual,
                Contributions::Residual,
            )?
            .iter()
            .zip(weighted_densities_entropy.iter())
            .map(|(s, w)| {
                (s * SIUnit::reference_volume().powi(-1)
                    / (&w.index_axis(Axis(0), 0) * SIUnit::reference_density()))
                // / (&w.slice(s![0, ..]) * SIUnit::reference_density()))
                .to_reduced(SIUnit::reference_moles().powi(-1))
                .unwrap()
            })
            .collect::<Vec<_>>();

        // for (i, edc) in self
        //     .dft
        //     .entropy_density_contributions::<Ix1>(
        //         temperature_red,
        //         &density_red,
        //         &convolver_dual,
        //         Contributions::Residual,
        //     )?
        //     .iter()
        //     .enumerate()
        // {
        //     let filename0 = i.to_string().to_owned();
        //     write_npy(filename0 + "_edc.npy", edc).unwrap();
        // }

        // for (i, entr_contrib) in entropy_molar_contributions.iter().enumerate() {
        //     let filename0 = i.to_string().to_owned();
        //     write_npy(filename0 + "_entr_contrib.npy", entr_contrib).unwrap();
        // }
        // ? what is this?
        // let mut dim = vec![];
        // self.density.shape().iter().skip(0).for_each(|&d| dim.push(d));
        // let mut entropy_molar = Array1::zeros(dim);//.into_dimensionality().unwrap();

        // sum the molar entropy of the individual contributions
        let mut entropy_molar = Array::zeros(self.density.raw_dim().remove_axis(Axis(0)));
        for contr in entropy_molar_contributions.iter() {
            entropy_molar += contr;
        }

        // residual entropy can not be larger than 0
        let s_res = entropy_molar.mapv(|s| f64::min(s, 0.0));

        // calculate the reference (Chapman-Enskog viscosity)
        let visc_ref = self
            .dft
            .functional
            .viscosity_reference::<Ix1>(&self.density, self.temperature)
            .unwrap();

        // write_npy(
        //     "visc_ref.npy",
        //     &visc_ref.to_reduced(SIUnit::reference_viscosity())?,
        // )
        // .unwrap();

        // let mut viscosity_shear = Array::zeros(entropy_molar.raw_dim());
        
        // calculate the shear viscosity from the residual reduced entropy 
        let mut viscosity_shear = self
            .dft
            .functional
            .viscosity_correlation::<Ix1>(&s_res, &self.density)
            .unwrap()
            .mapv(f64::exp)
            * visc_ref.to_reduced(SIUnit::reference_viscosity())?;

        // viscosity_shear.slice_mut(s![2..-2]).assign(
        //     &(self
        //         .dft
        //         .functional
        //         .viscosity_correlation::<Ix1>(&s_res, density)
        //         .unwrap()
        //         .mapv(f64::exp)
        //         * visc_ref.to_reduced(SIUnit::reference_viscosity())?),
        // );

        Ok(viscosity_shear * SIUnit::reference_viscosity())
    }

    // provide the weighted densities for entropy scaling
    // this function is only good for debugging
    pub fn weighted_densities_entropy(&self) -> EosResult<Vec<Array<f64, Ix2>>> {
        let temperature_red = self
            .temperature
            .to_reduced(SIUnit::reference_temperature())?;
        
        let functional_contributions_entropy = self
        .dft
        .functional
        .entropy_scaling_contributions();
        
        let weight_functions_entropy: Vec<WeightFunctionInfo<f64>> =
            functional_contributions_entropy
                .iter()
                .map(|c| c.weight_functions_entropy(temperature_red))
                .collect();
        let convolver_entropy: Rc<dyn Convolver<f64, Ix1>> =
            ConvolverFFT::plan(&self.grid, &weight_functions_entropy, None);

        let density_red = self.density.to_reduced(SIUnit::reference_density())?;

        Ok(convolver_entropy.weighted_densities(&density_red))
    }
}
