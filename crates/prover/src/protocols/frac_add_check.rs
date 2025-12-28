use binius_field::{Field, PackedField};
use binius_math::FieldBuffer;
use binius_utils::rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::protocols::intmul::witness;

/// Prover for the fractional addition protocol.
///
/// Each layer is a double of the numerator and denominator values of fractional terms. Each layer represents the addition of siblings with respect to the fractional addition rule:
/// $$\frac{a_0}{b_0} + \frac{a_1}{b_1} = \frac{a_0b_1 + a_1b_0}{b_0b_1}$
pub struct FracAddCheckProver<P: PackedField> {
	layers: Vec<(FieldBuffer<P>, FieldBuffer<P>)>,
}

pub type FractionalBuffer<P> = (FieldBuffer<P>, FieldBuffer<P>);
impl<F, P> FracAddCheckProver<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
    pub fn new(k: usize, witness: FractionalBuffer<P>)->(Self, FractionalBuffer<P>){
        let (witness_num, witness_den) = witness;
        assert_eq!(witness_num.log_len(), witness_den.log_len());
        assert!(k>= witness_num.log_len());

        let mut layers = Vec::with_capacity(k+1);
        layers.push((witness_num, witness_den));

       for _ in 0..k {
			let prev_layer = layers.last().expect("layers is non-empty");

            let (num, den) = prev_layer;
			let (num_0, num_1) = num
				.split_half_ref()
				.expect("layer has at least one variable");

			let (den_0, den_1) = den
				.split_half_ref()
				.expect("layer has at least one variable");

			let (next_layer_num, next_layer_den) = (num_0.as_ref(), den_0.as_ref(), num_1.as_ref(), den_1.as_ref())
				.into_par_iter()
				.map(|(&a_0, &b_0, &a_1, &b_1)| (a_0 * b_1 + a_1*b_0, b_0*b_1))
				.collect();

            let next_layer = (FieldBuffer::new(num.log_len() - 1, next_layer_num).expect("Should be half of previous layer"), FieldBuffer::new(den.log_len() - 1, next_layer_den).expect("Should be half of previous layer"));

			layers.push(next_layer);
		}

        let sums = layers.pop().expect("layers has k+1 elements"); 
        (Self { layers}, sums)
    }
}
