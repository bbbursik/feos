use crate::geometry::{Axis, Geometry, Grid};
use crate::weight_functions::*;
use ndarray::linalg::Dot;
use ndarray::prelude::*;
use ndarray::{Axis as Axis_nd, RemoveAxis, ScalarOperand, Slice};
use num_dual::*;
use num_traits::Zero;
use rustdct::DctNum;
use std::ops::{AddAssign, MulAssign, SubAssign};
use std::sync::Arc;

mod periodic_convolver;
mod transform;
pub use periodic_convolver::PeriodicConvolver;
use transform::*;

// use ndarray_npy::write_npy;

/// Trait for numerical convolutions for DFT.
///
/// Covers calculation of weighted densities & functional derivatives
/// from density profiles & profiles of the partial derivatives of the
/// Helmholtz energy functional.
///
/// Parametrized over data types `T` and dimension of the problem `D`.
pub trait Convolver<T, D: Dimension>: Send + Sync {
    /// Convolve the profile with the given weight function.
    fn convolve(&self, profile: Array<T, D>, weight_function: &WeightFunction<T>) -> Array<T, D>;

    /// Calculate weighted densities via convolution from density profiles.
    fn weighted_densities(&self, density: &Array<T, D::Larger>) -> Vec<Array<T, D::Larger>>;

    /// Calculate the functional derivative via convolution from partial derivatives
    /// of the Helmholtz energy functional.
    fn functional_derivative(
        &self,
        partial_derivatives: &[Array<T, D::Larger>],
        second_partial_derivatives: Option<&[Array<T, <D::Larger as Dimension>::Larger>]>,
        third_partial_derivatives: Option<
            &[Array<T, <<D::Larger as Dimension>::Larger as Dimension>::Larger>],
        >,
        density: Option<&Array<T, D::Larger>>,
    ) -> Array<T, D::Larger>;
}

// This is the gradient convolver for the local DFT approximation. It uses a second order Taylor Expansion for the weighted densities and a second order Taylor expansion of the partial derivatives of the Helmholtz energy density.
pub struct GradConvolver<T, D: Dimension> {
    grid: Array<T, D>,
    weight_functions: Vec<WeightFunctionInfo<T>>,
    weight_constants: Vec<Array<HyperDual64, D::Larger>>,
}

// This implementation has to be in Ix1 (1D) since the gradient would otherwise have to be calculated generally in 3D
impl<T> GradConvolver<T, Ix1>
where
    T: DctNum + DualNum<f64> + ScalarOperand,
{
    pub fn new(
        grid: Array<T, Ix1>,
        weight_functions: Vec<WeightFunctionInfo<T>>, //&[WeightFunctionInfo<T>],
        weight_constants: Vec<Array<HyperDual64, Ix2>>,
    ) -> Arc<dyn Convolver<T, Ix1>> {
        Arc::new(Self {
            grid,
            weight_functions,
            weight_constants,
        })
    }

    pub fn gradient(&self, f: &Array<T, Ix2>, dx: T) -> Array<T, Ix2> {
        let grad = Array::from_shape_fn(f.raw_dim(), |(c, i)| {
            let width: usize = 1;
            let d = if i as isize - width as isize <= 0 {
                (f[(c, i + width)] - f[(c, i)]) //* 0.0 // Left value --> where from?
            } else if i + width >= f.shape()[1] - 1 {
                (f[(c, i)] - f[(c, i - width)]) //* 0.0
            } else {
                f[(c, i + width)] - f[(c, i - width)]
                // (f[(c, i + 1)] - f[(c, i)]) * 2.0
            };
            d / (dx * 2.0 * width as f64)
        });
        grad
    }

    pub fn laplace(&self, f: &Array<T, Ix2>, dx: T) -> Array<T, Ix2> {
        let lapl = Array::from_shape_fn(f.raw_dim(), |(c, i)| {
            let width: usize = 1;
            let width_f64 = width as f64;
            let d = if i as isize - width as isize <= 0 {
                (f[(c, i + width * 2)] - f[(c, i + width)] * 2.0 + f[(c, i)]) //* 0.0
            } else if i + width >= f.shape()[1] - 1 {
                (f[(c, i)] - f[(c, i - width)] * 2.0 + f[(c, i - width * 2)]) // * 0.0
            } else {
                f[(c, i + width)] - f[(c, i)] * 2.0 + f[(c, i - width)]
            };

            d / (dx * dx) / (width_f64 * width_f64)
        });
        lapl
    }
}

// Implement the convolver Trait
impl<T> Convolver<T, Ix1> for GradConvolver<T, Ix1>
where
    T: DctNum + DualNum<f64> + ScalarOperand,
{
    fn convolve(
        &self,
        profile: Array<T, Ix1>,
        weight_function: &WeightFunction<T>,
    ) -> Array<T, Ix1> {
        unimplemented!();
    }

    fn weighted_densities(&self, density: &Array<T, Ix2>) -> Vec<Array<T, Ix2>> {
        let dx = self.grid[1] - self.grid[0];

        let gradient = self.gradient(density, dx);
        let laplace = self.laplace(density, dx);

        let k0 = HyperDual64::from(0.0).derive1().derive2();
        let mut weighted_densities_vec = Vec::with_capacity(self.weight_functions.len());

        let mut j = 0;
        for (wf, wc) in self
            .weight_functions
            .iter()
            .zip(self.weight_constants.iter())
        {
            let segments = wf.component_index.len();
            // let wf_hd = WeightFunctionInfo::from(HyperDual64::from(wf)); //brauche die wf mit T als Hyperdual für die weight constants, aber es kommt aus Interface als f64 --> übergeben aus Interface
            // let wf_hd = WeightFunctionInfo::from(*wf); //brauche die wf mit T als Hyperdual für die weight constants, aber es kommt aus Interface als f64 --> übergeben aus Interface
            // let w = wf.weight_constants(k0, 1); //can rewrite this with the corresponding function for the weight constants (i.e. scalar_weigth_const)
            let w0 = wc.mapv(|w| w.re);
            let w1 = wc.mapv(|w| -w.eps1[0]);
            let w2 = wc.mapv(|w| -0.5 * w.eps1eps2[(0, 0)]);

            // number of weighted densities
            let n_wd = wf.n_weighted_densities(1);

            // Allocating new array for intended weighted densities
            let mut dim = vec![n_wd];
            density.shape().iter().skip(1).for_each(|&d| dim.push(d));

            let mut weighted_densities = Array::zeros(dim).into_dimensionality().unwrap();

            // Initilaizing row index for non-local weighted densities
            let mut k = 0;

            // Assigning possible local densities to the front of the array
            if wf.local_density {
                weighted_densities
                    .slice_axis_mut(Axis_nd(0), Slice::from(0..segments))
                    .assign(&density);
                k += segments;
            }

            // Calculating weighted densities {scalar & vector, component}
            // only in 1D the vector and scalar wds can be calculated together like this
            // for _wf in &wf.scalar_component_weighted_densities {

            // Calculating functional derivative {scalar/vector, component}
            for _it in 0..(wf.scalar_component_weighted_densities.len()
                + wf.vector_component_weighted_densities.len())
            {
                for (i, (((rho, grad), lapl), mut res)) in density
                    .outer_iter()
                    .zip(gradient.outer_iter())
                    .zip(laplace.outer_iter())
                    .zip(
                        weighted_densities
                            .slice_axis_mut(Axis_nd(0), Slice::from(k..k + segments))
                            .outer_iter_mut(),
                    )
                    .enumerate()
                {
                    res.assign(
                        &(&rho * w0.slice(s![k..k + segments, ..]).into_diag()[i]
                            + &grad * w1.slice(s![k..k + segments, ..]).into_diag()[i]
                            + &lapl * w2.slice(s![k..k + segments, ..]).into_diag()[i]),
                    );
                }
                k += segments;
            }

            // for _wf in &wf.scalar_fmt_weighted_densities {

            // Calculating weighted densities {scalar/vector, FMT}
            // only in 1D the vector and scalar wds can be calculated together like this
            // Calculating functional derivative {scalar/vector, FMT}
            for _it in
                0..(wf.scalar_fmt_weighted_densities.len() + wf.vector_fmt_weighted_densities.len())
            {
                for (i, ((rho, grad), lapl)) in density
                    .outer_iter()
                    .zip(gradient.outer_iter())
                    .zip(laplace.outer_iter())
                    .enumerate()
                {
                    weighted_densities.index_axis_mut(Axis_nd(0), k).add_assign(
                        &(&rho * w0.slice(s![k, ..])[i]
                            + &grad * w1.slice(s![k, ..])[i]
                            + &lapl * w2.slice(s![k, ..])[i]),
                    );
                }
                k += 1;
            }

            weighted_densities_vec.push(weighted_densities);

            // man kann die wieghted densities für scalar und Vektor jeweils zusammenfassen da die w ja 0 sind
        }
        weighted_densities_vec
    }

    fn functional_derivative(
        &self,
        partial_derivatives: &[Array<T, Ix2>],
        second_partial_derivatives: Option<&[Array<T, Ix3>]>,
        third_partial_derivatives: Option<&[Array<T, Ix4>]>,
        density: Option<&Array<T, Ix2>>,
    ) -> Array<T, Ix2> {
        // Determine the dimension of the functional derivative, which is [n_segments, n_grid1, ngrid2, ... ]
        let mut dim = vec![(self.weight_functions[0]).component_index.len()];
        partial_derivatives[0]
            .shape()
            .iter()
            .skip(1)
            .for_each(|&d| dim.push(d));

        let dx = self.grid[1] - self.grid[0];

        // The gradients of the density are the same for all contributions and have to be calculated only once
        let gradient = self.gradient(density.unwrap(), dx);
        let laplace = self.laplace(density.unwrap(), dx);
        let gradient3 = self.gradient(&laplace, dx);
        let gradient4 = self.laplace(&laplace, dx);

        // Allocate arrays for individual terms of functional derivatives
        let mut functional_derivative_0: Array<T, Ix2> =
            Array::zeros(dim).into_dimensionality().unwrap();
        let mut functional_derivative_1: Array<T, Ix2> =
            Array::zeros(functional_derivative_0.raw_dim())
                .into_dimensionality()
                .unwrap();
        let mut functional_derivative_2: Array<T, Ix2> =
            Array::zeros(functional_derivative_0.raw_dim())
                .into_dimensionality()
                .unwrap();

        // loop through contributions
        for (i, ((((wf, wc), fpd), spd), tpd)) in self
            // for (i, (((wf, wc), pd), secparder)) in self
            .weight_functions
            .iter()
            .zip(self.weight_constants.iter())
            .zip(partial_derivatives.iter())
            .zip(second_partial_derivatives.unwrap().iter())
            .zip(third_partial_derivatives.unwrap().iter())
            .enumerate()
        {
            let n_segments = wf.component_index.len();
            let n_wd = fpd.shape()[0];
            let w0 = wc.mapv(|w| w.re);
            let w1 = wc.mapv(|w| -w.eps1[0]);
            let w2 = wc.mapv(|w| -0.5 * w.eps1eps2[(0, 0)]);

            let mut l = 0;

            // Assigning possible local densities to the front of the array
            if wf.local_density {
                functional_derivative_0 += &fpd.slice_axis(Axis_nd(0), Slice::from(..n_segments));
                l += n_segments;
            }

            // Calculating functional derivative {scalar/vector, component}
            for _it in 0..(wf.scalar_component_weighted_densities.len()
                + wf.vector_component_weighted_densities.len())
            {
                for i in 0..n_segments {
                    for alpha in 0..n_wd {
                        // 0-order term
                        functional_derivative_0
                            .index_axis_mut(Axis_nd(0), i)
                            .add_assign(
                                &(&fpd.index_axis(Axis_nd(0), alpha)
                                    * w0.slice(s![l..l + n_segments, ..]).into_diag()[i]),
                            );

                        for j in 0..n_segments {
                            for beta in 0..n_wd {
                                let grad_n_j_beta = &gradient.index_axis(Axis_nd(0), j)
                                    * w0.slice(s![l..l + n_segments, ..]).into_diag()[j]
                                    - &laplace.index_axis(Axis_nd(0), j)
                                        * w1.slice(s![l..l + n_segments, ..]).into_diag()[j]
                                    + &gradient3.index_axis(Axis_nd(0), j)
                                        * w2.slice(s![l..l + n_segments, ..]).into_diag()[j];
                                functional_derivative_1
                                    .index_axis_mut(Axis_nd(0), i)
                                    .add_assign(
                                        &(&spd
                                            .index_axis(Axis_nd(0), alpha)
                                            .index_axis(Axis_nd(0), beta)
                                            * &grad_n_j_beta
                                            * w1.slice(s![l..l + n_segments, ..]).into_diag()[i]),
                                    );

                                // \nabla^2 n for j,beta
                                let lapl_n = &laplace.index_axis(Axis_nd(0), j)
                                    * w0.slice(s![l..l + n_segments, ..]).into_diag()[j]
                                    - &gradient3.index_axis(Axis_nd(0), j)
                                        * w1.slice(s![l..l + n_segments, ..]).into_diag()[j]
                                    + &gradient4.index_axis(Axis_nd(0), j)
                                        * w2.slice(s![l..l + n_segments, ..]).into_diag()[j];

                                functional_derivative_2
                                    .index_axis_mut(Axis_nd(0), i)
                                    .add_assign(
                                        &(&spd
                                            .index_axis(Axis_nd(0), alpha)
                                            .index_axis(Axis_nd(0), beta)
                                            * lapl_n
                                            * w2.slice(s![l..l + n_segments, ..]).into_diag()[i]),
                                    );

                                for k in 0..n_segments {
                                    for gamma in 0..n_wd {
                                        let grad_n_gamma = &gradient.index_axis(Axis_nd(0), gamma)
                                            * w0.slice(s![l..l + n_segments, ..]).into_diag()[k]
                                            - &laplace.index_axis(Axis_nd(0), gamma)
                                                * w1.slice(s![l..l + n_segments, ..]).into_diag()
                                                    [k]
                                            + &gradient3.index_axis(Axis_nd(0), gamma)
                                                * w2.slice(s![l..l + n_segments, ..]).into_diag()
                                                    [k];

                                        functional_derivative_2
                                            .index_axis_mut(Axis_nd(0), i)
                                            .add_assign(
                                                &(&tpd
                                                    .index_axis(Axis_nd(0), alpha)
                                                    .index_axis(Axis_nd(0), beta)
                                                    .index_axis(Axis_nd(0), gamma)
                                                    * &grad_n_j_beta
                                                    * &grad_n_gamma
                                                    * w2.slice(s![l..l + n_segments, ..])
                                                        .into_diag()[i]),
                                            )
                                    }
                                }
                            }
                        }
                    }
                }
                l += n_segments;
            }

            // Calculating functional derivative {scalar/vector, FMT}
            for _it in 0..(wf.scalar_component_weighted_densities.len()
                + wf.vector_component_weighted_densities.len())
            {
                for i in 0..n_segments {
                    for alpha in 0..n_wd {
                        // 0-order term
                        functional_derivative_0
                            .index_axis_mut(Axis_nd(0), i)
                            .add_assign(
                                &(&fpd.index_axis(Axis_nd(0), alpha) * w0.slice(s![l, ..])[i]),
                            );

                        for j in 0..n_segments {
                            for beta in 0..n_wd {
                                let grad_n_j_beta = &gradient.index_axis(Axis_nd(0), j)
                                    * w0.slice(s![l, ..])[j]
                                    - &laplace.index_axis(Axis_nd(0), j) * w1.slice(s![l, ..])[j]
                                    + &gradient3.index_axis(Axis_nd(0), j) * w2.slice(s![l, ..])[j];
                                functional_derivative_1
                                    .index_axis_mut(Axis_nd(0), i)
                                    .add_assign(
                                        &(&spd
                                            .index_axis(Axis_nd(0), alpha)
                                            .index_axis(Axis_nd(0), beta)
                                            * &grad_n_j_beta
                                            * w1.slice(s![l, ..])[i]),
                                    );

                                // \nabla^2 n for j,beta
                                let lapl_n = &laplace.index_axis(Axis_nd(0), j)
                                    * w0.slice(s![l, ..])[j]
                                    - &gradient3.index_axis(Axis_nd(0), j) * w1.slice(s![l, ..])[j]
                                    + &gradient4.index_axis(Axis_nd(0), j) * w2.slice(s![l, ..])[j];

                                functional_derivative_2
                                    .index_axis_mut(Axis_nd(0), i)
                                    .add_assign(
                                        &(&spd
                                            .index_axis(Axis_nd(0), alpha)
                                            .index_axis(Axis_nd(0), beta)
                                            * lapl_n
                                            * w2.slice(s![l, ..])[i]),
                                    );

                                for k in 0..n_segments {
                                    for gamma in 0..n_wd {
                                        let grad_n_gamma = &gradient.index_axis(Axis_nd(0), gamma)
                                            * w0.slice(s![l, ..])[k]
                                            - &laplace.index_axis(Axis_nd(0), gamma)
                                                * w1.slice(s![l, ..])[k]
                                            + &gradient3.index_axis(Axis_nd(0), gamma)
                                                * w2.slice(s![l, ..])[k];

                                        functional_derivative_2
                                            .index_axis_mut(Axis_nd(0), i)
                                            .add_assign(
                                                &(&tpd
                                                    .index_axis(Axis_nd(0), alpha)
                                                    .index_axis(Axis_nd(0), beta)
                                                    .index_axis(Axis_nd(0), gamma)
                                                    * &grad_n_j_beta
                                                    * &grad_n_gamma
                                                    * w2.slice(s![l, ..])[i]),
                                            )
                                    }
                                }
                            }
                        }
                    }
                }
                l += n_segments;
            }
        }
        let functional_derivative =
            functional_derivative_0 + functional_derivative_1 + functional_derivative_2;
        functional_derivative
    }
}

pub(crate) struct BulkConvolver<T> {
    weight_constants: Vec<Array2<T>>,
}

impl<T: DualNum<f64>> BulkConvolver<T> {
    pub(crate) fn new(weight_functions: Vec<WeightFunctionInfo<T>>) -> Arc<dyn Convolver<T, Ix0>> {
        let weight_constants = weight_functions
            .into_iter()
            .map(|w| w.weight_constants(Zero::zero(), 0))
            .collect();
        Arc::new(Self { weight_constants })
    }
}

impl<T: DualNum<f64>> Convolver<T, Ix0> for BulkConvolver<T>
where
    Array2<T>: Dot<Array1<T>, Output = Array1<T>>,
{
    fn convolve(&self, _: Array0<T>, _: &WeightFunction<T>) -> Array0<T> {
        unreachable!()
    }

    fn weighted_densities(&self, density: &Array1<T>) -> Vec<Array1<T>> {
        self.weight_constants
            .iter()
            .map(|w| w.dot(density))
            .collect()
    }

    fn functional_derivative(
        &self,
        partial_derivatives: &[Array1<T>],
        second_partial_derivatives: Option<&[Array2<T>]>,
        third_partial_derivatives: Option<&[Array3<T>]>,
        density: Option<&Array1<T>>,
    ) -> Array1<T> {
        self.weight_constants
            .iter()
            .zip(partial_derivatives.iter())
            .map(|(w, pd)| pd.dot(w))
            .reduce(|a, b| a + b)
            .unwrap()
    }
}

/// Base structure to hold either information about the weight function through
/// `WeightFunctionInfo` or the weight functions themselves via
/// `FFTWeightFunctions`.
#[derive(Debug, Clone)]
struct FFTWeightFunctions<T, D: Dimension> {
    /// Either number of components for simple functionals
    /// or idividual segments for group contribution methods
    pub(crate) segments: usize,
    /// Flag if local density is required in the functional
    pub(crate) local_density: bool,
    /// Container for scalar component-wise weighted densities
    pub(crate) scalar_component_weighted_densities: Vec<Array<T, D::Larger>>,
    /// Container for vector component-wise weighted densities
    pub(crate) vector_component_weighted_densities: Vec<Array<T, <D::Larger as Dimension>::Larger>>,
    /// Container for scalar FMT weighted densities
    pub(crate) scalar_fmt_weighted_densities: Vec<Array<T, D::Larger>>,
    /// Container for vector FMT weighted densities
    pub(crate) vector_fmt_weighted_densities: Vec<Array<T, <D::Larger as Dimension>::Larger>>,
}

impl<T, D: Dimension> FFTWeightFunctions<T, D> {
    /// Calculates the total number of weighted densities for each functional
    /// from multiple weight functions depending on dimension.
    pub fn n_weighted_densities(&self, dimensions: usize) -> usize {
        (if self.local_density { self.segments } else { 0 })
            + self.scalar_component_weighted_densities.len() * self.segments
            + self.vector_component_weighted_densities.len() * self.segments * dimensions
            + self.scalar_fmt_weighted_densities.len()
            + self.vector_fmt_weighted_densities.len() * dimensions
    }
}

/// Convolver for 1-D, 2-D & 3-D systems using FFT algorithms to efficiently
/// compute convolutions in Fourier space.
///
/// Parametrized over the data type `T` and the dimension `D`.
#[derive(Clone)]
pub struct ConvolverFFT<T, D: Dimension> {
    /// k vectors
    k_abs: Array<f64, D>,
    /// Vector of weight functions for each component in multiple dimensions.
    weight_functions: Vec<FFTWeightFunctions<T, D>>,
    /// Lanczos sigma factor
    lanczos_sigma: Option<Array<f64, D>>,
    /// Possibly curvilinear Fourier transform in the first dimension
    transform: Arc<dyn FourierTransform<T>>,
    /// Vector of additional cartesian Fourier transforms in the other dimensions
    cartesian_transforms: Vec<Arc<CartesianTransform<T>>>,
}

impl<T, D: Dimension + RemoveAxis + 'static> ConvolverFFT<T, D>
where
    T: DctNum + DualNum<f64> + ScalarOperand,
    D::Larger: Dimension<Smaller = D>,
    D::Smaller: Dimension<Larger = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    /// Create the appropriate FFT convolver for the given grid.
    pub fn plan(
        grid: &Grid,
        weight_functions: &[WeightFunctionInfo<T>],
        lanczos: Option<i32>,
    ) -> Arc<dyn Convolver<T, D>> {
        match grid {
            Grid::Polar(r) => CurvilinearConvolver::new(r, &[], weight_functions, lanczos),
            Grid::Spherical(r) => CurvilinearConvolver::new(r, &[], weight_functions, lanczos),
            Grid::Cartesian1(z) => Self::new(Some(z), &[], weight_functions, lanczos),
            Grid::Cylindrical { r, z } => {
                CurvilinearConvolver::new(r, &[z], weight_functions, lanczos)
            }
            Grid::Cartesian2(x, y) => Self::new(Some(x), &[y], weight_functions, lanczos),
            Grid::Periodical2(x, y) => PeriodicConvolver::new(&[x, y], weight_functions, lanczos),
            Grid::Cartesian3(x, y, z) => Self::new(Some(x), &[y, z], weight_functions, lanczos),
            Grid::Periodical3(x, y, z) => {
                PeriodicConvolver::new(&[x, y, z], weight_functions, lanczos)
            }
        }
    }
}

impl<T, D: Dimension + 'static> ConvolverFFT<T, D>
where
    T: DctNum + DualNum<f64> + ScalarOperand,
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn new(
        axis: Option<&Axis>,
        cartesian_axes: &[&Axis],
        weight_functions: &[WeightFunctionInfo<T>],
        lanczos: Option<i32>,
    ) -> Arc<dyn Convolver<T, D>> {
        // initialize the Fourier transform
        let mut cartesian_transforms = Vec::with_capacity(cartesian_axes.len());
        let mut k_vec = Vec::with_capacity(cartesian_axes.len() + 1);
        let mut lengths = Vec::with_capacity(cartesian_axes.len() + 1);
        let (transform, k_x) = match axis {
            Some(axis) => match axis.geometry {
                Geometry::Cartesian => CartesianTransform::new(axis),
                Geometry::Cylindrical => PolarTransform::new(axis),
                Geometry::Spherical => SphericalTransform::new(axis),
            },
            None => NoTransform::new(),
        };
        k_vec.push(k_x);
        lengths.push(axis.map_or(1.0, |axis| axis.length()));
        for ax in cartesian_axes {
            let (transform, k_x) = CartesianTransform::new_cartesian(ax);
            cartesian_transforms.push(transform);
            k_vec.push(k_x);
            lengths.push(ax.length());
        }

        // Calculate the full k vectors
        let mut dim = vec![k_vec.len()];
        k_vec.iter().for_each(|k_x| dim.push(k_x.len()));
        let mut k: Array<_, D::Larger> = Array::zeros(dim).into_dimensionality().unwrap();
        let mut k_abs = Array::zeros(k.raw_dim().remove_axis(Axis(0)));
        for (i, (mut k_i, k_x)) in k.outer_iter_mut().zip(k_vec.iter()).enumerate() {
            k_i.lanes_mut(Axis_nd(i))
                .into_iter()
                .for_each(|mut l| l.assign(k_x));
            k_abs.add_assign(&k_i.mapv(|k| k.powi(2)));
        }
        k_abs.map_inplace(|k| *k = k.sqrt());

        // Lanczos sigma factor
        let lanczos_sigma = lanczos.map(|exp| {
            let mut lanczos = Array::ones(k_abs.raw_dim());
            for (i, (k_x, &l)) in k_vec.iter().zip(lengths.iter()).enumerate() {
                let points = k_x.len();
                let m2 = if points % 2 == 0 {
                    points as f64 + 2.0
                } else {
                    points as f64 + 1.0
                };
                let l_x = k_x.mapv(|k| (k * l / m2).sph_j0().powi(exp));
                for mut l in lanczos.lanes_mut(Axis_nd(i)) {
                    l.mul_assign(&l_x);
                }
            }
            lanczos
        });

        // calculate weight functions in Fourier space and weight constants
        let mut fft_weight_functions = Vec::with_capacity(weight_functions.len());
        for wf in weight_functions {
            // Calculates the weight functions values from `k_abs`
            // Pre-allocation of empty `Vec`
            let mut scal_comp = Vec::with_capacity(wf.scalar_component_weighted_densities.len());
            // Filling array with scalar component-wise weight functions
            for wf_i in &wf.scalar_component_weighted_densities {
                scal_comp.push(wf_i.fft_scalar_weight_functions(&k_abs, &lanczos_sigma));
            }

            // Pre-allocation of empty `Vec`
            let mut vec_comp = Vec::with_capacity(wf.vector_component_weighted_densities.len());
            // Filling array with vector-valued component-wise weight functions
            for wf_i in &wf.vector_component_weighted_densities {
                vec_comp.push(wf_i.fft_vector_weight_functions(&k_abs, &k, &lanczos_sigma));
            }

            // Pre-allocation of empty `Vec`
            let mut scal_fmt = Vec::with_capacity(wf.scalar_fmt_weighted_densities.len());
            // Filling array with scalar FMT weight functions
            for wf_i in &wf.scalar_fmt_weighted_densities {
                scal_fmt.push(wf_i.fft_scalar_weight_functions(&k_abs, &lanczos_sigma));
            }

            // Pre-allocation of empty `Vec`
            let mut vec_fmt = Vec::with_capacity(wf.vector_fmt_weighted_densities.len());
            // Filling array with vector-valued FMT weight functions
            for wf_i in &wf.vector_fmt_weighted_densities {
                vec_fmt.push(wf_i.fft_vector_weight_functions(&k_abs, &k, &lanczos_sigma));
            }

            // Initializing `FFTWeightFunctions` structure
            fft_weight_functions.push(FFTWeightFunctions::<_, D> {
                segments: wf.component_index.len(),
                local_density: wf.local_density,
                scalar_component_weighted_densities: scal_comp,
                vector_component_weighted_densities: vec_comp,
                scalar_fmt_weighted_densities: scal_fmt,
                vector_fmt_weighted_densities: vec_fmt,
            });
        }

        // Return `FFTConvolver<T, D>`
        Arc::new(Self {
            k_abs,
            weight_functions: fft_weight_functions,
            lanczos_sigma,
            transform,
            cartesian_transforms,
        })
    }
}

impl<T, D: Dimension> ConvolverFFT<T, D>
where
    T: DctNum + DualNum<f64> + ScalarOperand,
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn forward_transform(&self, f: ArrayView<T, D>, vector_index: Option<usize>) -> Array<T, D> {
        let mut dim = vec![self.k_abs.shape()[0]];
        f.shape().iter().skip(1).for_each(|&d| dim.push(d));
        let mut result: Array<_, D> = Array::zeros(dim.clone()).into_dimensionality().unwrap();
        for (f, r) in f
            .lanes(Axis_nd(0))
            .into_iter()
            .zip(result.lanes_mut(Axis_nd(0)))
        {
            self.transform
                .forward_transform(f, r, vector_index.map_or(true, |ind| ind != 0));
        }
        for (i, transform) in self.cartesian_transforms.iter().enumerate() {
            dim[i + 1] = self.k_abs.shape()[i + 1];
            let mut res: Array<_, D> = Array::zeros(dim.clone()).into_dimensionality().unwrap();
            for (f, r) in result
                .lanes(Axis_nd(i + 1))
                .into_iter()
                .zip(res.lanes_mut(Axis_nd(i + 1)))
            {
                transform.forward_transform(f, r, vector_index.map_or(true, |ind| ind != i + 1));
            }
            result = res;
        }

        result
    }

    fn forward_transform_comps(
        &self,
        f: ArrayView<T, D::Larger>,
        vector_index: Option<usize>,
    ) -> Array<T, D::Larger> {
        let mut dim = vec![f.shape()[0]];
        self.k_abs.shape().iter().for_each(|&d| dim.push(d));
        let mut result = Array::zeros(dim).into_dimensionality().unwrap();
        for (f, mut r) in f.outer_iter().zip(result.outer_iter_mut()) {
            r.assign(&self.forward_transform(f, vector_index));
        }
        result
    }

    fn back_transform(
        &self,
        mut f: ArrayViewMut<T, D>,
        mut result: ArrayViewMut<T, D>,
        vector_index: Option<usize>,
    ) {
        let mut dim = vec![result.shape()[0]];
        f.shape().iter().skip(1).for_each(|&d| dim.push(d));
        let mut res: Array<_, D> = Array::zeros(dim.clone()).into_dimensionality().unwrap();
        for (f, r) in f
            .lanes_mut(Axis_nd(0))
            .into_iter()
            .zip(res.lanes_mut(Axis_nd(0)))
        {
            self.transform
                .back_transform(f, r, vector_index.map_or(true, |ind| ind != 0));
        }
        for (i, transform) in self.cartesian_transforms.iter().enumerate() {
            dim[i + 1] = result.shape()[i + 1];
            let mut res2: Array<_, D> = Array::zeros(dim.clone()).into_dimensionality().unwrap();
            for (f, r) in res
                .lanes_mut(Axis_nd(i + 1))
                .into_iter()
                .zip(res2.lanes_mut(Axis_nd(i + 1)))
            {
                transform.back_transform(f, r, vector_index.map_or(true, |ind| ind != i + 1));
            }
            res = res2;
        }

        result.assign(&res);
    }

    fn back_transform_comps(
        &self,
        mut f: Array<T, D::Larger>,
        mut result: ArrayViewMut<T, D::Larger>,
        vector_index: Option<usize>,
    ) {
        for (f, r) in f.outer_iter_mut().zip(result.outer_iter_mut()) {
            self.back_transform(f, r, vector_index);
        }
    }
}

impl<T, D: Dimension> Convolver<T, D> for ConvolverFFT<T, D>
where
    T: DctNum + ScalarOperand + DualNum<f64>,
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn convolve(&self, profile: Array<T, D>, weight_function: &WeightFunction<T>) -> Array<T, D> {
        // Forward transform
        let f_k = self.forward_transform(profile.view(), None);

        // calculate weight function
        let w = weight_function
            .fft_scalar_weight_functions(&self.k_abs, &self.lanczos_sigma)
            .index_axis_move(Axis(0), 0);

        // Inverse transform
        let mut result = Array::zeros(profile.raw_dim());
        self.back_transform((f_k * w).view_mut(), result.view_mut(), None);
        result
    }

    fn weighted_densities(&self, density: &Array<T, D::Larger>) -> Vec<Array<T, D::Larger>> {
        // Applying FFT to each row of the matrix `rho` saving the result in `rho_k`
        let rho_k = self.forward_transform_comps(density.view(), None);

        // Iterate over all contributions
        let mut weighted_densities_vec = Vec::with_capacity(self.weight_functions.len());
        for wf in &self.weight_functions {
            // number of weighted densities
            let n_wd = wf.n_weighted_densities(density.ndim() - 1);

            // Allocating new array for intended weighted densities
            let mut dim = vec![n_wd];
            density.shape().iter().skip(1).for_each(|&d| dim.push(d));
            let mut weighted_densities = Array::zeros(dim).into_dimensionality().unwrap();

            // Initilaizing row index for non-local weighted densities
            let mut k = 0;

            // Assigning possible local densities to the front of the array
            if wf.local_density {
                weighted_densities
                    .slice_axis_mut(Axis(0), Slice::from(0..wf.segments))
                    .assign(density);
                k += wf.segments;
            }

            // Calculating weighted densities {scalar, component}
            for wf_i in &wf.scalar_component_weighted_densities {
                self.back_transform_comps(
                    &rho_k * wf_i,
                    weighted_densities.slice_axis_mut(Axis(0), Slice::from(k..k + wf.segments)),
                    None,
                );
                k += wf.segments;
            }

            // Calculating weighted densities {vector, component}
            for wf_i in &wf.vector_component_weighted_densities {
                for (i, wf_i) in wf_i.outer_iter().enumerate() {
                    self.back_transform_comps(
                        &rho_k * &wf_i,
                        weighted_densities.slice_axis_mut(Axis(0), Slice::from(k..k + wf.segments)),
                        Some(i),
                    );
                    k += wf.segments;
                }
            }

            // Calculating weighted densities {scalar, FMT}
            for wf_i in &wf.scalar_fmt_weighted_densities {
                self.back_transform(
                    (&rho_k * wf_i).sum_axis(Axis(0)).view_mut(),
                    weighted_densities.index_axis_mut(Axis(0), k),
                    None,
                );
                k += 1;
            }

            // Calculating weighted densities {vector, FMT}
            for wf_i in &wf.vector_fmt_weighted_densities {
                for (i, wf_i) in wf_i.outer_iter().enumerate() {
                    self.back_transform(
                        (&rho_k * &wf_i).sum_axis(Axis(0)).view_mut(),
                        weighted_densities.index_axis_mut(Axis(0), k),
                        Some(i),
                    );
                    k += 1;
                }
            }

            // add weighted densities for this contribution to the result
            weighted_densities_vec.push(weighted_densities);
        }
        // Return
        weighted_densities_vec
    }

    fn functional_derivative(
        &self,
        partial_derivatives: &[Array<T, D::Larger>],
        second_partial_derivatives: Option<&[Array<T, <D::Larger as Dimension>::Larger>]>,
        third_partial_derivatives: Option<
            &[Array<T, <<D::Larger as Dimension>::Larger as Dimension>::Larger>],
        >,
        density: Option<&Array<T, D::Larger>>,
    ) -> Array<T, D::Larger> {
        // Allocate arrays for the result, the local contribution to the functional derivative,
        // the functional derivative in Fourier space, and the bulk contributions
        let mut dim = vec![self.weight_functions[0].segments];
        partial_derivatives[0]
            .shape()
            .iter()
            .skip(1)
            .for_each(|&d| dim.push(d));
        let mut functional_deriv = Array::zeros(dim).into_dimensionality().unwrap();
        let mut functional_deriv_local = Array::zeros(functional_deriv.raw_dim());
        let mut dim = vec![self.weight_functions[0].segments];
        self.k_abs.shape().iter().for_each(|&d| dim.push(d));
        let mut functional_deriv_k = Array::zeros(dim).into_dimensionality().unwrap();

        // Iterate over all contributions
        for (pd, wf) in partial_derivatives.iter().zip(&self.weight_functions) {
            // Multiplication of `partial_derivatives` with the weight functions in
            // Fourier space (convolution in real space); summation leads to
            // functional derivative: the rows in the array are selected from the
            // running variable `k` with the number of rows needed for this
            // particular contribution
            let mut k = 0;

            // If local densities are present, their contributions are added directly
            if wf.local_density {
                functional_deriv_local += &pd.slice_axis(Axis(0), Slice::from(..wf.segments));
                k += wf.segments;
            }

            // Convolution of functional derivatives {scalar, component}
            for wf_i in &wf.scalar_component_weighted_densities {
                let pd_k = self.forward_transform_comps(
                    pd.slice_axis(Axis(0), Slice::from(k..k + wf.segments)),
                    None,
                );
                functional_deriv_k.add_assign(&(&pd_k * wf_i));
                k += wf.segments;
            }

            // Convolution of functional derivatives {vector, component}
            for wf_i in &wf.vector_component_weighted_densities {
                for (i, wf_i) in wf_i.outer_iter().enumerate() {
                    let pd_k = self.forward_transform_comps(
                        pd.slice_axis(Axis(0), Slice::from(k..k + wf.segments)),
                        Some(i),
                    );
                    functional_deriv_k.add_assign(&(pd_k * &wf_i));
                    k += wf.segments;
                }
            }

            // Convolution of functional derivatives {scalar, FMT}
            for wf_i in &wf.scalar_fmt_weighted_densities {
                let pd_k = self.forward_transform(pd.index_axis(Axis(0), k), None);
                functional_deriv_k.add_assign(&(wf_i * &pd_k));
                k += 1;
            }

            // Convolution of functional derivatives {vector, FMT}
            for wf_i in &wf.vector_fmt_weighted_densities {
                for (i, wf_i) in wf_i.outer_iter().enumerate() {
                    let pd_k = self.forward_transform(pd.index_axis(Axis(0), k), Some(i));
                    functional_deriv_k.add_assign(&(&wf_i * &pd_k));
                    k += 1;
                }
            }
        }

        // Backward transform of the non-local part of the functional derivative
        self.back_transform_comps(functional_deriv_k, functional_deriv.view_mut(), None);

        // Return sum over non-local and local contributions
        functional_deriv + functional_deriv_local
    }
}

/// The curvilinear convolver accounts for the shift that has to be performed
/// for spherical and polar transforms.
struct CurvilinearConvolver<T, D> {
    convolver: Arc<dyn Convolver<T, D>>,
    convolver_boundary: Arc<dyn Convolver<T, D>>,
}

impl<T, D: Dimension + RemoveAxis + 'static> CurvilinearConvolver<T, D>
where
    T: DctNum + ScalarOperand + DualNum<f64>,
    D::Larger: Dimension<Smaller = D>,
    D::Smaller: Dimension<Larger = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn new(
        r: &Axis,
        z: &[&Axis],
        weight_functions: &[WeightFunctionInfo<T>],
        lanczos: Option<i32>,
    ) -> Arc<dyn Convolver<T, D>> {
        Arc::new(Self {
            convolver: ConvolverFFT::new(Some(r), z, weight_functions, lanczos),
            convolver_boundary: ConvolverFFT::new(None, z, weight_functions, lanczos),
        })
    }
}

impl<T, D: Dimension + RemoveAxis> Convolver<T, D> for CurvilinearConvolver<T, D>
where
    T: DctNum + ScalarOperand + DualNum<f64>,
    D::Smaller: Dimension<Larger = D>,
    D::Larger: Dimension<Smaller = D>,
{
    fn convolve(
        &self,
        mut profile: Array<T, D>,
        weight_function: &WeightFunction<T>,
    ) -> Array<T, D> {
        // subtract boundary profile from full profile
        let profile_boundary = profile
            .index_axis(Axis(0), profile.shape()[0] - 1)
            .into_owned();
        for mut lane in profile.outer_iter_mut() {
            lane.sub_assign(&profile_boundary);
        }

        // convolve full profile
        let mut result = self.convolver.convolve(profile, weight_function);

        // convolve boundary profile
        let profile_boundary = profile_boundary.insert_axis(Axis(0));
        let result_boundary = self
            .convolver_boundary
            .convolve(profile_boundary, weight_function);

        // Add boundary result back to full result
        let result_boundary = result_boundary.index_axis(Axis(0), 0);
        for mut lane in result.outer_iter_mut() {
            lane.add_assign(&result_boundary);
        }
        result
    }

    /// Calculates weighted densities via convolution from density profiles.
    fn weighted_densities(&self, density: &Array<T, D::Larger>) -> Vec<Array<T, D::Larger>> {
        // subtract boundary profile from full profile
        let density_boundary = density.index_axis(Axis(1), density.shape()[1] - 1);
        let mut density = density.to_owned();
        for mut lane in density.axis_iter_mut(Axis(1)) {
            lane.sub_assign(&density_boundary);
        }

        // convolve full profile
        let mut wd = self.convolver.weighted_densities(&density);

        // convolve boundary profile
        let density_boundary = density_boundary.insert_axis(Axis(1));
        let wd_boundary = self
            .convolver_boundary
            .weighted_densities(&density_boundary.to_owned());

        // Add boundary result back to full result
        for (wd, wd_boundary) in wd.iter_mut().zip(wd_boundary.iter()) {
            let wd_view = wd_boundary.index_axis(Axis(1), 0);
            for mut lane in wd.axis_iter_mut(Axis(1)) {
                lane.add_assign(&wd_view);
            }
        }

        wd
    }

    /// Calculates the functional derivative via convolution from partial derivatives
    /// of the Helmholtz energy functional.
    fn functional_derivative(
        &self,
        partial_derivatives: &[Array<T, D::Larger>],
        second_partial_derivatives: Option<&[Array<T, <D::Larger as Dimension>::Larger>]>,
        third_partial_derivatives: Option<
            &[Array<T, <<D::Larger as Dimension>::Larger as Dimension>::Larger>],
        >,
        density: Option<&Array<T, D::Larger>>,
    ) -> Array<T, D::Larger> {
        // subtract boundary profile from full profile
        let mut partial_derivatives_full = Vec::new();
        let mut partial_derivatives_boundary = Vec::new();
        for pd in partial_derivatives {
            let pd_boundary = pd.index_axis(Axis(1), pd.shape()[1] - 1).to_owned();
            let mut pd_full = pd.to_owned();
            for mut lane in pd_full.axis_iter_mut(Axis(1)) {
                lane.sub_assign(&pd_boundary);
            }
            partial_derivatives_full.push(pd_full);
            partial_derivatives_boundary.push(pd_boundary);
        }

        // convolve full profile
        let mut functional_derivative =
            self.convolver
                .functional_derivative(&partial_derivatives_full, None, None, None);

        // convolve boundary profile
        let mut partial_derivatives_boundary = Vec::new();
        for pd in partial_derivatives {
            let mut pd_boundary = pd.view();
            pd_boundary.collapse_axis(Axis(1), pd.shape()[1] - 1);
            partial_derivatives_boundary.push(pd_boundary.to_owned());
        }
        let functional_derivative_boundary = self.convolver_boundary.functional_derivative(
            &partial_derivatives_boundary,
            None,
            None,
            None,
        );

        // Add boundary result back to full result
        let functional_derivative_view = functional_derivative_boundary.index_axis(Axis(1), 0);
        for mut lane in functional_derivative.axis_iter_mut(Axis(1)) {
            lane.add_assign(&functional_derivative_view);
        }

        functional_derivative
    }
}
