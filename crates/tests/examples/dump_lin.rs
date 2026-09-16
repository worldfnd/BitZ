//! Single-column opening dump (f2z-pcs feasibility probe): the shape the
//! end-to-end proof hands the PCS after `transpose_query` — a
//! `LinearClaim<F128>` over `Shape::new(22, 0)` (2^22 row weights, column
//! weight one) on 2^22 bits committed under the (14, 8) reference split —
//! proved with `Pcs::prove_lin(.., StatementBinding::Bind, ..)` under the
//! dump labels, verified here, and written out for `bitz_lin_probe`.
//!
//! Usage: `dump_lin <seed> <out-dir>`
use std::io::Write;

use common::{LinearClaim, OpeningQuery, Shape};
use field::F128;
use num_traits::ConstZero;
use pcs::{CommitScheme, HashKind, LigeritoProfile, Pcs, StatementBinding};
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use tests::{prover_transcript, verifier_transcript};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args[1].parse().unwrap();
    let out = std::path::PathBuf::from(&args[2]);
    std::fs::create_dir_all(&out).unwrap();

    let committed_shape = Shape::new(14, 8).unwrap();
    let query_shape = Shape::new(22, 0).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let packed: Vec<F128> = (0..(1usize << committed_shape.log_bits()) / 128)
        .map(|_| F128::new(rng.next_u64(), rng.next_u64()))
        .collect();
    let pcs = Pcs::new(&committed_shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let (root, data) = pcs.commit(&packed).unwrap();

    let weights: Vec<F128> = (0..query_shape.rows())
        .map(|_| F128::new(rng.next_u64(), rng.next_u64()))
        .collect();
    let mut target = F128::ZERO;
    for (i, weight) in weights.iter().enumerate() {
        let element = packed[i / 128];
        let bit = if i % 128 < 64 {
            element.lo >> (i % 128) & 1
        } else {
            element.hi >> (i % 128 - 64) & 1
        };
        if bit == 1 {
            target += *weight;
        }
    }
    let claim =
        LinearClaim::from_shape(&query_shape, weights, vec![F128::from(1u64)], target).unwrap();
    let query = OpeningQuery::InnerProduct { claim };

    let started = std::time::Instant::now();
    let mut transcript = prover_transcript();
    pcs.prove_lin(&data, packed.clone(), &query, StatementBinding::Bind, &mut transcript)
        .expect("prove_lin");
    let proof = transcript.finish();
    println!("prove: {:.1?}", started.elapsed());
    let started = std::time::Instant::now();
    let mut verifier = verifier_transcript(&proof);
    pcs.verify_lin(&root, &query, StatementBinding::Bind, &mut verifier)
        .expect("verify_lin");
    verifier.check_eof().expect("eof");
    println!("verify: {:.1?}", started.elapsed());

    let mut w = std::fs::File::create(out.join("witness.bin")).unwrap();
    for e in &packed {
        w.write_all(&e.lo.to_le_bytes()).unwrap();
        w.write_all(&e.hi.to_le_bytes()).unwrap();
    }
    let OpeningQuery::InnerProduct { claim } = &query else {
        unreachable!()
    };
    let mut c = std::io::BufWriter::new(std::fs::File::create(out.join("claim.bin")).unwrap());
    for weights in [claim.row_weights(), claim.column_weights()] {
        c.write_all(&(weights.len() as u64).to_le_bytes()).unwrap();
        for weight in weights {
            c.write_all(&weight.to_bytes()).unwrap();
        }
    }
    c.write_all(&claim.target().to_bytes()).unwrap();
    drop(c);
    std::fs::write(out.join("narg.bin"), &proof.narg_string).unwrap();
    std::fs::write(out.join("hints.bin"), &proof.hints).unwrap();
    let meta = format!(
        "t={}\ns={}\nm={}\ncommitted_t={}\ncommitted_s={}\nseed={seed}\nroot={}\nsession=bitz-tests\ninstance=fold-round-trip\nnarg_len={}\nhints_len={}\n",
        query_shape.log_rows(),
        query_shape.log_columns(),
        query_shape.log_bits(),
        committed_shape.log_rows(),
        committed_shape.log_columns(),
        hex(&root.0),
        proof.narg_string.len(),
        proof.hints.len(),
    );
    std::fs::write(out.join("meta.txt"), &meta).unwrap();
    print!("{meta}");
}
