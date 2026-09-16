use common::{BitZParams, LinearClaim, Root, Shape};
use field::Fq;
use transcript::Encoding;

use crate::{HashKind, Pcs};

impl Pcs {
    /// Encodes the public bit claim, commitment geometry, and resolved PCS configuration.
    /// A map digest selects virtual mode; direct mode omits it. Integers use little-endian
    /// encoding, and configuration vectors carry a u64 element count.
    pub fn bitz_statement<const Q: u128>(
        &self,
        root: &Root,
        params: &BitZParams<Q>,
        committed_shape: Shape,
        map_digest: Option<&[u8; 32]>,
        claim: &LinearClaim<Fq<Q>>,
    ) -> Vec<u8> {
        let mut frame = b"bitz/statement/v1".to_vec();
        frame.push(u8::from(map_digest.is_some()));
        frame.extend_from_slice(&root.0);
        frame.extend_from_slice(params.encode().as_ref());
        let config = self.prover_config();
        let put = |frame: &mut Vec<u8>, value: usize| {
            frame.extend_from_slice(&(value as u64).to_le_bytes());
        };
        for value in [
            committed_shape.log_rows(),
            committed_shape.log_columns(),
            self.params().m,
            config.recursive_steps,
            config.initial_log_msg_cols,
            config.initial_log_num_interleaved,
            config.initial_k,
        ] {
            put(&mut frame, value);
        }
        for values in [
            &config.log_inv_rates,
            &config.recursive_log_msg_cols,
            &config.recursive_ks,
            &config.queries,
            &config.grinding_bits,
            &config.fold_grinding_bits,
            &config.ood_samples,
        ] {
            put(&mut frame, values.len());
            for &value in values {
                put(&mut frame, value);
            }
        }
        frame.push(match config.merkle_hash {
            HashKind::Sha256 => 0,
            HashKind::Blake3 => 1,
        });
        if let Some(digest) = map_digest {
            frame.extend_from_slice(digest);
        }
        frame.extend_from_slice(claim.encode().as_ref());
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LigeritoProfile;
    use field::{F128, gf128::smallest_generator};

    #[test]
    fn every_statement_input_changes_the_first_challenge() {
        const Q: u128 = (1 << 114) - 11;
        let shape = Shape::new(9, 13).unwrap();
        let params = BitZParams::<Q>::new(shape, smallest_generator()).unwrap();
        let pcs = Pcs::new(&shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
        let root = Root([7; 32]);
        let rows = vec![Fq::from(2u128); shape.rows()];
        let columns = vec![Fq::from(3u128); shape.columns()];
        let claim =
            LinearClaim::new(&params, rows.clone(), columns.clone(), Fq::from(5u128)).unwrap();
        let frame = pcs.bitz_statement(&root, &params, shape, Some(&[11; 32]), &claim);
        assert!(frame.starts_with(b"bitz/statement/v1"));
        let challenge = |bytes: &[u8]| {
            let mut prover = transcript::build_prover(b"binding-test", b"instance");
            prover.public_message(bytes);
            let result: F128 = prover.verifier_message();
            let proof = transcript::Proof::default();
            let mut verifier = transcript::build_verifier(b"binding-test", b"instance", &proof);
            verifier.public_message(bytes);
            assert_eq!(result, verifier.verifier_message::<F128>());
            result
        };
        let baseline = challenge(&frame);
        let changed = |bytes: Vec<u8>| {
            assert_ne!(frame, bytes);
            assert_ne!(baseline, challenge(&bytes));
        };
        changed(pcs.bitz_statement(&Root([8; 32]), &params, shape, Some(&[11; 32]), &claim));
        changed(pcs.bitz_statement(
            &root,
            &params,
            Shape::new(10, 12).unwrap(),
            Some(&[11; 32]),
            &claim,
        ));
        changed(pcs.bitz_statement(&root, &params, shape, Some(&[12; 32]), &claim));
        changed(pcs.bitz_statement(&root, &params, shape, None, &claim));
        let other_params =
            BitZParams::<Q>::new(Shape::new(10, 12).unwrap(), smallest_generator()).unwrap();
        changed(pcs.bitz_statement(&root, &other_params, shape, Some(&[11; 32]), &claim));
        let generator = smallest_generator();
        let other_params = BitZParams::<Q>::new(shape, generator * generator).unwrap();
        changed(pcs.bitz_statement(&root, &other_params, shape, Some(&[11; 32]), &claim));
        let other_params = BitZParams::<97>::new(shape, generator).unwrap();
        let other_claim = LinearClaim::new(
            &other_params,
            vec![Fq::from(2u128); shape.rows()],
            vec![Fq::from(3u128); shape.columns()],
            Fq::from(5u128),
        )
        .unwrap();
        changed(pcs.bitz_statement(&root, &other_params, shape, Some(&[11; 32]), &other_claim));
        for (index, mut values) in [(0, rows.clone()), (1, columns.clone())] {
            values[0] += Fq::ONE;
            let other = if index == 0 {
                LinearClaim::new(&params, values, columns.clone(), claim.target())
            } else {
                LinearClaim::new(&params, rows.clone(), values, claim.target())
            }
            .unwrap();
            changed(pcs.bitz_statement(&root, &params, shape, Some(&[11; 32]), &other));
        }
        let other = LinearClaim::new(&params, rows, columns, Fq::from(6u128)).unwrap();
        changed(pcs.bitz_statement(&root, &params, shape, Some(&[11; 32]), &other));
        for (profile, hash) in [
            (LigeritoProfile::Slim, HashKind::Blake3),
            (LigeritoProfile::Fast, HashKind::Sha256),
        ] {
            let other = Pcs::new(&shape, profile, hash).unwrap();
            changed(other.bitz_statement(&root, &params, shape, Some(&[11; 32]), &claim));
        }
    }
}
