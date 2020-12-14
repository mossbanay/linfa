use linfa::{dataset::Dataset, dataset::Pr, dataset::Targets, traits::Fit, traits::Predict};
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Data, Ix2};
use std::cmp::Ordering;

use super::permutable_kernel::{Kernel, PermutableKernel, PermutableKernelOneClass};
use super::solver_smo::SolverState;
use super::SolverParams;
use super::{Float, Svm, SvmParams};

/// Support Vector Classification with C-penalizing parameter
///
/// This methods solves a binary SVC problem with a penalizing parameter C between (0, inf). The
/// dual problem has the form
/// ```ignore
/// min_a 1/2*a^tQ a - e^T a s.t. y^t = 0, 0 <= a_i <= C_i
/// ```
/// with `Q_ij = y_i y_j K(x_i, x_j)` the kernel matrix.
///
/// # Parameters
///
/// * `params` - Solver parameters (threshold etc.)
/// * `kernel` - the kernel matrix `Q`
/// * `targets` - the ground truth targets `y_i`
/// * `cpos` - C for positive targets
/// * `cneg` - C for negative targets
pub fn fit_c<'a, A: Float>(
    params: SolverParams<A>,
    kernel: &'a Kernel<'a, A>,
    targets: &'a [bool],
    cpos: A,
    cneg: A,
) -> Svm<'a, A, Pr> {
    let bounds = targets
        .iter()
        .map(|x| if *x { cpos } else { cneg })
        .collect::<Vec<_>>();

    let kernel = PermutableKernel::new(kernel, targets.to_vec());

    let solver = SolverState::new(
        vec![A::zero(); targets.len()],
        vec![-A::one(); targets.len()],
        targets.to_vec(),
        kernel,
        bounds,
        params,
        false,
    );

    let mut res = solver.solve();

    res.alpha = res
        .alpha
        .into_iter()
        .zip(targets.iter())
        .map(|(a, b)| if *b { a } else { -a })
        .collect();

    res.with_phantom()
}

/// Support Vector Classification with Nu-penalizing term
///
/// This methods solves a binary SVC problem with a penalizing parameter nu between (0, 1). The
/// dual problem has the form
/// ```ignore
/// min_a 1/2*a^tQ a s.t. y^t a = 0, 0 <= a_i <= 1/l, e^t a > nu
/// ```
/// with `Q_ij = y_i y_j K(x_i, x_j)` the kernel matrix.
///
/// # Parameters
///
/// * `params` - Solver parameters (threshold etc.)
/// * `kernel` - the kernel matrix `Q`
/// * `targets` - the ground truth targets `y_i`
/// * `nu` - Nu penalizing term
pub fn fit_nu<'a, A: Float>(
    params: SolverParams<A>,
    kernel: &'a Kernel<'a, A>,
    targets: &'a [bool],
    nu: A,
) -> Svm<'a, A, Pr> {
    let mut sum_pos = nu * A::from(targets.len()).unwrap() / A::from(2.0).unwrap();
    let mut sum_neg = nu * A::from(targets.len()).unwrap() / A::from(2.0).unwrap();
    let init_alpha = targets
        .iter()
        .map(|x| {
            if *x {
                let val = A::min(A::one(), sum_pos);
                sum_pos -= val;
                val
            } else {
                let val = A::min(A::one(), sum_neg);
                sum_neg -= val;
                val
            }
        })
        .collect::<Vec<_>>();

    let kernel = PermutableKernel::new(kernel, targets.to_vec());

    let solver = SolverState::new(
        init_alpha,
        vec![A::zero(); targets.len()],
        targets.to_vec(),
        kernel,
        vec![A::one(); targets.len()],
        params,
        true,
    );

    let mut res = solver.solve();

    let r = res.r.unwrap();

    res.alpha = res
        .alpha
        .into_iter()
        .zip(targets.iter())
        .map(|(a, b)| if *b { a } else { -a })
        .map(|x| x / r)
        .collect();
    res.rho /= r;
    res.obj /= r * r;

    res.with_phantom()
}

/// Support Vector Classification for one-class problems
///
/// This methods solves a binary SVC, when there are no targets available. This can, for example be
/// useful, when outliers should be rejected.
///
/// # Parameters
///
/// * `params` - Solver parameters (threshold etc.)
/// * `kernel` - the kernel matrix `Q`
/// * `nu` - Nu penalizing term
pub fn fit_one_class<'a, A: Float + num_traits::ToPrimitive>(
    params: SolverParams<A>,
    kernel: &'a Kernel<'a, A>,
    nu: A,
) -> Svm<'a, A, Pr> {
    let size = kernel.size();
    let n = (nu * A::from(size).unwrap()).to_usize().unwrap();

    let init_alpha = (0..size)
        .map(|x| match x.cmp(&n) {
            Ordering::Less => A::one(),
            Ordering::Greater => A::zero(),
            Ordering::Equal => nu * A::from(size).unwrap() - A::from(x).unwrap(),
        })
        .collect::<Vec<_>>();

    let kernel = PermutableKernelOneClass::new(kernel);

    let solver = SolverState::new(
        init_alpha,
        vec![A::zero(); size],
        vec![true; size],
        kernel,
        vec![A::one(); size],
        params,
        false,
    );

    let res = solver.solve();

    res.with_phantom()
}

impl<'a, F: Float> Fit<'a, Kernel<'a, F>, Vec<bool>> for SvmParams<F, Pr> {
    type Object = Svm<'a, F, Pr>;

    fn fit(&self, dataset: &'a Dataset<Kernel<'a, F>, Vec<bool>>) -> Self::Object {
        match (self.c, self.nu) {
            (Some((c_p, c_n)), _) => fit_c(
                self.solver_params.clone(),
                &dataset.records,
                dataset.targets(),
                c_p,
                c_n,
            ),
            (None, Some((nu, _))) => fit_nu(
                self.solver_params.clone(),
                &dataset.records,
                dataset.targets(),
                nu,
            ),
            _ => panic!("Set either C value or Nu value"),
        }
    }
}

impl<'a, F: Float> Fit<'a, Kernel<'a, F>, &Vec<bool>> for SvmParams<F, Pr> {
    type Object = Svm<'a, F, Pr>;

    fn fit(&self, dataset: &'a Dataset<Kernel<'a, F>, &Vec<bool>>) -> Self::Object {
        match (self.c, self.nu) {
            (Some((c_p, c_n)), _) => fit_c(
                self.solver_params.clone(),
                &dataset.records,
                dataset.targets(),
                c_p,
                c_n,
            ),
            (None, Some((nu, _))) => fit_nu(
                self.solver_params.clone(),
                &dataset.records,
                dataset.targets(),
                nu,
            ),
            _ => panic!("Set either C value or Nu value"),
        }
    }
}

impl<'a, F: Float> Fit<'a, Kernel<'a, F>, &[bool]> for SvmParams<F, Pr> {
    type Object = Svm<'a, F, Pr>;

    fn fit(&self, dataset: &'a Dataset<Kernel<'a, F>, &[bool]>) -> Self::Object {
        match (self.c, self.nu) {
            (Some((c_p, c_n)), _) => fit_c(
                self.solver_params.clone(),
                &dataset.records,
                dataset.targets(),
                c_p,
                c_n,
            ),
            (None, Some((nu, _))) => fit_nu(
                self.solver_params.clone(),
                &dataset.records,
                dataset.targets(),
                nu,
            ),
            _ => panic!("Set either C value or Nu value"),
        }
    }
}
impl<'a, F: Float> Fit<'a, Kernel<'a, F>, &()> for SvmParams<F, Pr> {
    type Object = Svm<'a, F, Pr>;

    fn fit(&self, dataset: &'a Dataset<Kernel<'a, F>, &()>) -> Self::Object {
        match self.nu {
            Some((nu, _)) => fit_one_class(self.solver_params.clone(), &dataset.records, nu),
            None => panic!("One class needs Nu value"),
        }
    }
}

/// Predict a probability with a feature vector
impl<'a, F: Float> Predict<Array1<F>, Pr> for Svm<'a, F, Pr> {
    fn predict(&self, data: Array1<F>) -> Pr {
        let val = match self.linear_decision {
            Some(ref x) => x.dot(&data) - self.rho,
            None => self.kernel.weighted_sum(&self.alpha, data.view()) - self.rho,
        };

        // this is safe because `F` is only implemented for `f32` and `f64`
        Pr(val.to_f32().unwrap())
    }
}

/// Predict a probability with a set of observations
impl<'a, F: Float, D: Data<Elem = F>> Predict<ArrayBase<D, Ix2>, Vec<Pr>> for Svm<'a, F, Pr> {
    fn predict(&self, data: ArrayBase<D, Ix2>) -> Vec<Pr> {
        data.outer_iter()
            .map(|data| {
                let val = match self.linear_decision {
                    Some(ref x) => x.dot(&data) - self.rho,
                    None => self.kernel.weighted_sum(&self.alpha, data.view()) - self.rho,
                };

                // this is safe because `F` is only implemented for `f32` and `f64`
                Pr(val.to_f32().unwrap())
            })
            .collect()
    }
}

impl<'a, F: Float, T: Targets> Predict<Dataset<Array2<F>, T>, Dataset<Array2<F>, Vec<Pr>>>
    for Svm<'a, F, Pr>
{
    fn predict(&self, data: Dataset<Array2<F>, T>) -> Dataset<Array2<F>, Vec<Pr>> {
        let Dataset { records, .. } = data;
        let predicted = self.predict(records.view());

        Dataset::new(records, predicted)
    }
}

impl<'a, F: Float, T: Targets, D: Data<Elem = F>>
    Predict<&'a Dataset<ArrayBase<D, Ix2>, T>, Dataset<ArrayView2<'a, F>, Vec<Pr>>>
    for Svm<'a, F, Pr>
{
    fn predict(
        &self,
        data: &'a Dataset<ArrayBase<D, Ix2>, T>,
    ) -> Dataset<ArrayView2<'a, F>, Vec<Pr>> {
        let predicted = self.predict(data.records.view());

        Dataset::new(data.records.view(), predicted)
    }
}

#[cfg(test)]
mod tests {
    use super::Svm;
    use linfa::dataset::Dataset;
    use linfa::metrics::ToConfusionMatrix;
    use linfa::traits::{Fit, Predict, Transformer};
    use linfa_kernel::{Kernel, KernelMethod};

    use ndarray::{Array, Array2, Axis};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand_isaac::Isaac64Rng;

    pub fn generate_convoluted_rings(n_points: usize) -> Array2<f64> {
        let mut out = Array::random((n_points * 2, 2), Uniform::new(0f64, 1.));
        for (i, mut elm) in out.outer_iter_mut().enumerate() {
            // generate convoluted rings with 1/10th noise
            let phi = 6.28 * elm[1];
            let eps = elm[0] / 10.0;

            if i < n_points {
                elm[0] = 1.0 * phi.cos() + eps;
                elm[1] = 1.0 * phi.sin() + eps;
            } else {
                elm[0] = 5.0 * phi.cos() + eps;
                elm[1] = 5.0 * phi.sin() + eps;
            }
        }

        out
    }

    #[test]
    fn test_linear_classification() {
        let entries = ndarray::stack(
            Axis(0),
            &[
                Array::random((10, 2), Uniform::new(-1., -0.5)).view(),
                Array::random((10, 2), Uniform::new(0.5, 1.)).view(),
            ],
        )
        .unwrap();
        let targets = (0..20).map(|x| x < 10).collect::<Vec<_>>();
        let dataset = Dataset::new(entries.clone(), targets);

        let dataset = Kernel::params()
            .method(KernelMethod::Linear)
            .transform(&dataset);

        // train model with positive and negative weight
        let model = Svm::params().pos_neg_weights(1.0, 1.0).fit(&dataset);

        let valid = model
            .predict(Dataset::from(entries))
            .map_targets(|x| **x > 0.0);

        let cm = valid.confusion_matrix(&dataset);
        assert_eq!(cm.accuracy(), 1.0);

        // train model with Nu parameter
        let model = Svm::params().nu_weight(0.05).fit(&dataset);

        let valid = model.predict(valid).map_targets(|x| **x > 0.0);

        let cm = valid.confusion_matrix(&dataset);
        assert_eq!(cm.accuracy(), 1.0);
    }

    #[test]
    fn test_polynomial_classification() {
        let mut rng = Isaac64Rng::seed_from_u64(42);
        // construct parabolica and classify middle area as positive and borders as negative
        let records = Array::random_using((40, 1), Uniform::new(-2f64, 2.), &mut rng);
        let targets = records.map_axis(Axis(1), |x| x[0] * x[0] < 0.5).to_vec();
        let dataset = Dataset::new(records.clone(), targets);

        let dataset = Kernel::params()
            .method(KernelMethod::Polynomial(0.0, 2.0))
            .transform(&dataset);

        // train model with positive and negative weight
        let model = Svm::params().pos_neg_weights(1.0, 1.0).fit(&dataset);

        //println!("{:?}", model.predict(Dataset::from(records.clone())).targets());

        let valid = model
            .predict(Dataset::from(records))
            .map_targets(|x| **x > 0.0);

        let cm = valid.confusion_matrix(&dataset);
        assert!(cm.accuracy() > 0.9);
    }

    #[test]
    fn test_convoluted_rings_classification() {
        let records = generate_convoluted_rings(10);
        let targets = (0..20).map(|x| x < 10).collect::<Vec<_>>();
        let dataset = Dataset::new(records.clone(), targets);

        let dataset = Kernel::params()
            .method(KernelMethod::Gaussian(50.0))
            .transform(&dataset);

        // train model with positive and negative weight
        let model = Svm::params().pos_neg_weights(1.0, 1.0).fit(&dataset);

        let valid = model
            .predict(Dataset::from(records))
            .map_targets(|x| **x > 0.0);

        let cm = valid.confusion_matrix(&dataset);
        assert!(cm.accuracy() > 0.9);

        // train model with Nu parameter
        let model = Svm::params().nu_weight(0.01).fit(&dataset);

        let valid = model.predict(valid).map_targets(|x| **x > 0.0);

        let cm = valid.confusion_matrix(&dataset);
        assert!(cm.accuracy() > 0.9);
    }

    #[test]
    fn test_reject_classification() {
        // generate two clusters with 100 samples each
        let entries = Array::random((100, 2), Uniform::new(-4., 4.));
        let dataset = Dataset::new(entries.clone(), ());

        let dataset = Kernel::params()
            .method(KernelMethod::Gaussian(100.0))
            .transform(&dataset);

        // train model with positive and negative weight
        let model = Svm::params().nu_weight(1.0).fit(&dataset);

        let valid = Dataset::from(Array::random((100, 2), Uniform::new(-10., 10f32)));
        let valid = model.predict(valid).map_targets(|x| **x > 0.0);

        // count the number of correctly rejected samples
        let mut rejected = 0;
        let mut total = 0;
        for (pred, pos) in valid.targets().iter().zip(valid.records.outer_iter()) {
            let distance = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
            if distance >= 5.0 {
                if !pred {
                    rejected += 1;
                }
                total += 1;
            }
        }

        // at least 95% should be correctly rejected
        assert!((rejected as f32) / (total as f32) > 0.95);
    }
}
