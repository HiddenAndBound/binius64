// Copyright 2025-2026 The Binius Developers

use std::{collections::VecDeque, iter::zip};

use binius_field::{Field, PackedField};
use binius_math::{
	AsSlicesMut, FieldBuffer, FieldSliceMut, multilinear::fold::fold_highest_var_inplace,
	univariate::evaluate_univariate,
};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use binius_verifier::protocols::sumcheck::RoundCoeffs;
use itertools::{Itertools, izip};

use crate::protocols::sumcheck::{
	Error, common::SumcheckProver, gruen32::Gruen32, round_evals::RoundEvals2,
};

// Batch quadratic mle check prover for M compositions/claims of N multilinears.
pub struct BatchQuadraticMleCheckProver<
	P: PackedField,
	Composition,
	InfinityComposition,
	const N: usize,
	const M: usize,
> {
	multilinears: Box<dyn AsSlicesMut<P, N> + Send>,
	composition: Composition,
	infinity_composition: InfinityComposition,
	last_coeffs_or_eval: RoundCoeffsOrEvals<P::Scalar, M>,
	gruen32: Gruen32<P>,
}

impl<F, P, Composition, InfinityComposition, const N: usize, const M: usize>
	BatchQuadraticMleCheckProver<P, Composition, InfinityComposition, N, M>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Composition: Fn([P; N], P, &mut [P; M]) + Sync,
	InfinityComposition: Fn([P; N], P, &mut [P; M]) + Sync,
{
	pub fn new(
		mut multilinears: impl AsSlicesMut<P, N> + Send + 'static,
		composition: Composition,
		infinity_composition: InfinityComposition,
		eval_point: Vec<F>,
		eval_claims: [F; M],
	) -> Result<Self, Error> {
		let n_vars = eval_point.len();
		assert!(N > 0 && M > 0);
		for multilinear in &multilinears.as_slices_mut() {
			if multilinear.log_len() != n_vars {
				return Err(Error::MultilinearSizeMismatch);
			}
		}

		let last_coeffs_or_eval = RoundCoeffsOrEvals::Evals(eval_claims);
		let gruen32 = Gruen32::new(&eval_point);

		Ok(Self {
			multilinears: Box::new(multilinears),
			composition,
			infinity_composition,
			last_coeffs_or_eval,
			gruen32,
		})
	}

	/// Gets mutable slices of the multilinears, truncated to the current number of variables.
	fn multilinears_mut(&mut self) -> [FieldSliceMut<'_, P>; N] {
		let n_vars = self.gruen32.n_vars_remaining();
		let mut slices = self.multilinears.as_slices_mut();
		for slice in &mut slices {
			slice.truncate(n_vars);
		}
		slices
	}
}

impl<F, P, Composition, InfinityComposition, const N: usize, const M: usize> SumcheckProver<F>
	for BatchQuadraticMleCheckProver<P, Composition, InfinityComposition, N, M>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Composition: Fn([P; N], P, &mut [P; M]) + Sync,
	InfinityComposition: Fn([P; N], P, &mut [P; M]) + Sync,
{
	fn n_vars(&self) -> usize {
		self.gruen32.n_vars_remaining()
	}

	fn n_claims(&self) -> usize {
		M
	}

	fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, Error> {
		let last_eval = match &self.last_coeffs_or_eval {
			RoundCoeffsOrEvals::Evals(evals) => *evals,
			RoundCoeffsOrEvals::Coeffs(_) => return Err(Error::ExpectedFold),
		};

		let n_vars_remaining = self.gruen32.n_vars_remaining();
		assert!(n_vars_remaining > 0);

		let eq_expansion = self.gruen32.eq_expansion();
		assert_eq!(eq_expansion.log_len(), n_vars_remaining - 1);

		let comp = &self.composition;
		let inf_comp = &self.infinity_composition;

		// Get multilinear slices and truncate to current n_vars
		let mut multilinears = self.multilinears.as_slices_mut();
		for slice in &mut multilinears {
			slice.truncate(n_vars_remaining);
		}
		let (splits_0, splits_1) = multilinears
			.iter()
			.map(FieldBuffer::split_half_ref)
			.collect::<Result<(Vec<_>, Vec<_>), _>>()?;

		let round_evals = [[P::zero(); M]; 2];

		//TODO: Chunked compute like in frac_add and bivariate multimle
		let partial_sums = eq_expansion
			.as_ref()
			.into_par_iter()
			.enumerate()
			.fold(
				|| round_evals,
				|[mut y_1, mut y_inf], (i, &eq_i)| {
					let mut evals_1 = [P::default(); N];
					let mut evals_inf = [P::default(); N];

					izip!(&splits_0, &splits_1, &mut evals_1, &mut evals_inf).for_each(
						|(lo, hi, eval_1, eval_inf)| {
							*eval_1 = lo.as_ref()[i];
							*eval_inf = hi.as_ref()[i];
						},
					);

					comp(evals_1, eq_i, &mut y_1);
					inf_comp(evals_inf, eq_i, &mut y_inf);
					[y_1, y_inf]
				},
			)
			.collect::<Vec<_>>();

		let packed_round_evals = partial_sums
			.into_iter()
			.map(|[coeffs_1, coeffs_inf]| {
				zip(coeffs_1, coeffs_inf)
					.map(|(y_1, y_inf)| RoundEvals2 { y_1, y_inf })
					.collect::<Vec<_>>()
			})
			.reduce(|acc, g| zip(acc, g).map(|(acc_i, g_i)| acc_i + &g_i).collect())
			.expect("Will be non_empty");

		let alpha = self.gruen32.next_coordinate();
		let round_coeffs = packed_round_evals
			.into_iter()
			.zip_eq(last_eval)
			.map(|(packed_evals, sum)| {
				let round_evals = packed_evals.sum_scalars(n_vars_remaining);
				round_evals.interpolate_eq(sum, alpha)
			})
			.collect::<Vec<_>>();
		self.last_coeffs_or_eval = RoundCoeffsOrEvals::Coeffs(
			round_coeffs
				.clone()
				.try_into()
				.expect("Will have M elements."),
		);
		Ok(round_coeffs)
	}

	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		let RoundCoeffsOrEvals::Coeffs(prime_coeffs) = &self.last_coeffs_or_eval else {
			return Err(Error::ExpectedExecute);
		};

		assert!(
			self.n_vars() > 0,
			"n_vars is decremented in fold; \
			fold changes last_coeffs_or_eval to Eval variant; \
			fold only executes with Coeffs variant; \
			thus, n_vars should be > 0"
		);

		let evals = prime_coeffs
			.iter()
			.map(|coeffs| coeffs.evaluate(challenge))
			.collect_array()
			.expect("Will have size M");

		for multilinear in &mut self.multilinears_mut() {
			fold_highest_var_inplace(multilinear, challenge)?;
		}

		self.gruen32.fold(challenge)?;
		self.last_coeffs_or_eval = RoundCoeffsOrEvals::Evals(evals);
		Ok(())
	}

	fn finish(mut self) -> Result<Vec<F>, Error> {
		if self.n_vars() > 0 {
			let error = match self.last_coeffs_or_eval {
				RoundCoeffsOrEvals::Coeffs(_) => Error::ExpectedFold,
				RoundCoeffsOrEvals::Evals(_) => Error::ExpectedExecute,
			};

			return Err(error);
		}

		let multilinear_evals = self
			.multilinears_mut()
			.into_iter()
			.map(|multilinear| multilinear.get_checked(0).expect("multilinear.len() == 1"))
			.collect();

		Ok(multilinear_evals)
	}
}

#[derive(Debug, Clone)]
enum RoundCoeffsOrEvals<F: Field, const M: usize> {
	Coeffs([RoundCoeffs<F>; M]),
	Evals([F; M]),
}
