use super::{FluidParameters, PoreProfile, PoreSpecification};
use crate::{
    Axis, ConvolverFFT, DFTProfile, DFTSpecifications, Grid, HelmholtzEnergyFunctional, DFT,
};
use feos_core::{EosResult, EosUnit, State};
use ndarray::{Array3, Ix2};
use quantity::si::{SIArray3, SINumber, SIUnit};

pub struct Pore2D {
    system_size: [SINumber; 2],
    n_grid: [usize; 2],
}

pub type PoreProfile2D<F> = PoreProfile<Ix2, F>;

impl Pore2D {
    pub fn new(system_size: [SINumber; 2], n_grid: [usize; 2]) -> Self {
        Self {
            system_size,
            n_grid,
        }
    }
}

impl PoreSpecification<Ix2> for Pore2D {
    fn initialize<F: HelmholtzEnergyFunctional + FluidParameters>(
        &self,
        bulk: &State<DFT<F>>,
        density: Option<&SIArray3>,
        external_potential: Option<&Array3<f64>>,
    ) -> EosResult<PoreProfile<Ix2, F>> {
        let dft: &F = &bulk.eos;

        // generate grid
        let x = Axis::new_cartesian(self.n_grid[0], self.system_size[0], None)?;
        let y = Axis::new_cartesian(self.n_grid[1], self.system_size[1], None)?;

        // temperature
        let t = bulk
            .temperature
            .to_reduced(SIUnit::reference_temperature())?;

        // initialize convolver
        let grid = Grid::Cartesian2(x, y);
        let weight_functions = dft.weight_functions(t);
        let convolver = ConvolverFFT::plan(&grid, &weight_functions, Some(1));

        /////////////////////////
        // Initialize DFTProfile
        let mut profile =
            DFTProfile::new(grid, convolver, bulk, external_potential.cloned(), density)?;

        // specify the specification
        profile.specification = DFTSpecifications::total_moles_from_profile(&profile)?;

        Ok(PoreProfile {
            // profile: DFTProfile::new(grid, convolver, bulk, external_potential.cloned(), density)?,
            profile: profile,
            grand_potential: None,
            interfacial_tension: None,
        })
    }

    fn dimension(&self) -> i32 {
        2
    }
}
