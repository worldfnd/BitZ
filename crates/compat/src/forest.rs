//! Binary product forest and its equality-factored sumchecks.
use crate::messages::{BinaryFields, binary, binary_value};
use crate::statement::Bitified;
use field::{F128 as F, FixedBasePow};
use num_traits::{ConstOne, ConstZero, Inv};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::io;
use transcript::reference::{Bytes, Described, Transcript, number};

pub fn integer(v: F) -> u128 {
    v.lo as u128 | ((v.hi as u128) << 64)
}
pub fn element(v: u128) -> F {
    F::new(v as u64, (v >> 64) as u64)
}
pub fn eq_table(point: &[F]) -> Vec<F> {
    let mut v = vec![F::ONE];
    for &r in point {
        let n = v.len();
        v.resize(n * 2, F::ZERO);
        for i in 0..n {
            let right = v[i] * r;
            v[n + i] = right;
            v[i] += right;
        }
    }
    v
}
pub fn eq(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(F::ONE, |p, (&x, &y)| p * (F::ONE + x + y))
}
pub fn fold(v: &mut Vec<F>, r: F) {
    for i in 0..v.len() / 2 {
        v[i] = v[2 * i] + r * (v[2 * i] + v[2 * i + 1]);
    }
    v.truncate(v.len() / 2);
}
pub fn evaluate(v: &[F], point: &[F]) -> F {
    let mut v = v.to_vec();
    for &r in point {
        fold(&mut v, r);
    }
    assert_eq!(v.len(), 1);
    v[0]
}
fn draw(t: &mut Transcript, purpose: &str, ctx: Value) -> io::Result<F> {
    if purpose == "sumcheck.round_challenge" {
        binary(t, purpose, ctx, true).map(element)
    } else {
        binary(t, purpose, ctx, false).map(element)
    }
}
pub fn draw_point(t: &mut Transcript, purpose: &str, n: usize) -> io::Result<Vec<F>> {
    (0..n)
        .map(|coordinate| draw(t, purpose, json!({"coordinate":coordinate})))
        .collect()
}
fn fields(t: &mut Transcript, purpose: &str, ctx: Value, v: &[F]) -> io::Result<()> {
    t.absorb(
        purpose,
        ctx,
        &BinaryFields(&v.iter().copied().map(integer).collect::<Vec<_>>()),
    )
}
pub fn tagged(t: &mut Transcript, tag: u8, values: &[F], ctx: Value) -> io::Result<()> {
    let mut bytes = vec![tag];
    for v in values {
        bytes.extend_from_slice(&integer(*v).to_le_bytes());
    }
    let (purpose, value) = match tag {
        0x30 => (
            "gkr.product_tree_roots",
            json!({"field":"GF(2^128)","basis":"polynomial","roots_hex":values.iter().map(|&v|number(integer(v))).collect::<Vec<_>>()}),
        ),
        0x20 => (
            "ring_switch.evaluation_vector",
            json!({"field":"GF(2^128)","basis":"polynomial","evaluations_hex":values.iter().map(|&v|number(integer(v))).collect::<Vec<_>>()}),
        ),
        0x32 => (
            "gkr.closing_child_pair",
            json!({"wire_tag":tag,"values":values.iter().map(|&v|binary_value(integer(v))).collect::<Vec<_>>()}),
        ),
        _ => unreachable!(),
    };
    t.absorb(
        purpose,
        ctx,
        &Described {
            encoding: Bytes(&bytes),
            display: &value,
        },
    )
}
fn header(t: &mut Transcript, n: usize, layer: Option<usize>) -> io::Result<()> {
    let (ctx, values, display) = if let Some(layer) = layer {
        (
            json!({"layer":layer}),
            vec![n as u128, 3],
            json!({"variables":n,"degree_bound":3}),
        )
    } else {
        (
            json!({}),
            vec![n as u128, 1, 2],
            json!({"variables":n,"group_count":1,"degree_bounds":[2]}),
        )
    };
    t.absorb(
        "sumcheck.header",
        ctx,
        &Described {
            encoding: BinaryFields(&values),
            display: &display,
        },
    )
}
#[derive(Clone)]
pub struct Sumcheck {
    pub sum: F,
    pub messages: Vec<[F; 2]>,
}
#[derive(Clone)]
pub struct Layer {
    pub x: Option<Sumcheck>,
    pub c: Sumcheck,
    pub pair: [F; 2],
}
#[derive(Clone)]
pub struct Proof {
    pub folds: Vec<u128>,
    pub layers: Vec<Layer>,
    pub presum: Sumcheck,
}
struct Group {
    left: Vec<F>,
    right: Vec<F>,
    scale: F,
}
fn prove_eq(
    t: &mut Transcript,
    mut groups: Vec<Group>,
    q: &[F],
    layer: usize,
) -> io::Result<(Sumcheck, Vec<F>, Vec<[F; 2]>)> {
    header(t, q.len(), Some(layer))?;
    let mut proof = Sumcheck {
        sum: F::ZERO,
        messages: vec![],
    };
    let mut point = vec![];
    for round in 0..q.len() {
        let weights = eq_table(&q[round + 1..]);
        let [c0, c1, c2] = groups
            .par_iter()
            .map(|g| {
                let mut c = [F::ZERO; 3];
                for (i, &weight) in weights.iter().enumerate() {
                    let a = g.left[2 * i];
                    let b = g.left[2 * i + 1] + a;
                    let x = g.right[2 * i];
                    let y = g.right[2 * i + 1] + x;
                    c[0] += weight * (a * x);
                    c[1] += weight * (a * y + b * x);
                    c[2] += weight * (b * y);
                }
                c.map(|x| x * g.scale)
            })
            .reduce(
                || [F::ZERO; 3],
                |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
            );
        if round == 0 {
            proof.sum = c0 + q[round] * (c1 + c2);
        }
        fields(
            t,
            "sumcheck.round_polynomial",
            json!({"layer":layer,"round":round}),
            &[c1, c2],
        )?;
        let r = draw(
            t,
            "sumcheck.round_challenge",
            json!({"layer":layer,"round":round}),
        )?;
        proof.messages.push([c1, c2]);
        point.push(r);
        let e = F::ONE + q[round] + r;
        groups.par_iter_mut().for_each(|g| {
            g.scale *= e;
            fold(&mut g.left, r);
            fold(&mut g.right, r);
        });
    }
    Ok((
        proof,
        point,
        groups.iter().map(|g| [g.left[0], g.right[0]]).collect(),
    ))
}
fn verify_eq(t: &mut Transcript, p: &Sumcheck, q: &[F], layer: usize) -> io::Result<(Vec<F>, F)> {
    if p.messages.len() != q.len() {
        return Err(io::Error::other("forest round count"));
    }
    header(t, q.len(), Some(layer))?;
    let mut claim = p.sum;
    let mut point = vec![];
    for (round, &[c1, c2]) in p.messages.iter().enumerate() {
        fields(
            t,
            "sumcheck.round_polynomial",
            json!({"layer":layer,"round":round}),
            &[c1, c2],
        )?;
        let r = draw(
            t,
            "sumcheck.round_challenge",
            json!({"layer":layer,"round":round}),
        )?;
        let c0 = claim + q[round] * (c1 + c2);
        claim = (F::ONE + q[round] + r) * (c0 + r * (c1 + r * c2));
        point.push(r);
    }
    Ok((point, claim))
}
fn product_coefficients(a: &[F], b: &[F]) -> [F; 3] {
    let mut c = [F::ZERO; 3];
    for i in 0..a.len() / 2 {
        let x = a[2 * i];
        let dx = x + a[2 * i + 1];
        let y = b[2 * i];
        let dy = y + b[2 * i + 1];
        c[0] += x * y;
        c[1] += x * dy + dx * y;
        c[2] += dx * dy;
    }
    c
}
fn prove_presum(
    t: &mut Transcript,
    mut a: Vec<F>,
    mut b: Vec<F>,
) -> io::Result<(Sumcheck, Vec<F>, F)> {
    header(t, 15, None)?;
    let mut p = Sumcheck {
        sum: F::ZERO,
        messages: vec![],
    };
    let mut point = vec![];
    let two = F::from(2u64);
    for round in 0..15 {
        let [c0, c1, c2] = product_coefficients(&a, &b);
        if round == 0 {
            p.sum = c1 + c2;
        }
        let tail = [c0 + c1 + c2, c0 + two * c1 + two * two * c2];
        fields(
            t,
            "sumcheck.round_polynomial",
            json!({"group":0,"round":round}),
            &tail,
        )?;
        let r = draw(t, "sumcheck.round_challenge", json!({"round":round}))?;
        p.messages.push(tail);
        point.push(r);
        fold(&mut a, r);
        fold(&mut b, r);
    }
    Ok((p, point, b[0]))
}
fn verify_presum(t: &mut Transcript, p: &Sumcheck) -> io::Result<(Vec<F>, F)> {
    if p.messages.len() != 15 {
        return Err(io::Error::other("presum rounds"));
    }
    header(t, 15, None)?;
    let two = F::from(2u64);
    let inv = (two * two + two).inv().unwrap();
    let mut claim = p.sum;
    let mut point = vec![];
    for (round, &tail) in p.messages.iter().enumerate() {
        fields(
            t,
            "sumcheck.round_polynomial",
            json!({"round":round,"group":0}),
            &tail,
        )?;
        let r = draw(t, "sumcheck.round_challenge", json!({"round":round}))?;
        let c0 = claim + tail[0];
        let c2 = (tail[1] + c0 + two * claim) * inv;
        let c1 = claim + c2;
        claim = c0 + r * (c1 + r * c2);
        point.push(r);
    }
    Ok((point, claim))
}
pub fn prove(
    t: &mut Transcript,
    packed: &[F],
    opening: &Bitified,
) -> io::Result<(Proof, Vec<F>, F)> {
    let table = common::BitTable::new(common::Shape::new(15, 7).unwrap(), packed)
        .map_err(|e| io::Error::other(format!("{e:?}")))?;
    let comb = FixedBasePow::new(F::from(2u64), 8);
    let powers = common::row_images(&comb, &opening.rows);
    let folds = common::fold_columns(&table, &opening.rows);
    let roots = common::column_images(&comb, &folds);
    // Adjacent *halves* collapse the highest remaining row coordinate.
    let trees: Vec<Vec<Vec<F>>> = (0..128)
        .into_par_iter()
        .map(|c| {
            let mut levels = vec![
                (0..32768)
                    .map(|r| if table.bit(c, r) { powers[r] } else { F::ONE })
                    .collect::<Vec<_>>(),
            ];
            while levels.last().unwrap().len() > 1 {
                let v = levels.last().unwrap();
                let n = v.len() / 2;
                levels.push((0..n).map(|i| v[i] * v[n + i]).collect());
            }
            levels.reverse();
            levels
        })
        .collect();
    for c in 0..128 {
        assert_eq!(trees[c][0][0], roots[c]);
    }
    tagged(t, 0x30, &roots, json!({}))?;
    let mut zc = draw_point(t, "gkr.tree_batching_point", 7)?;
    let mut zx = vec![];
    let mut layers = vec![];
    for layer in 0..15 {
        let eqc = eq_table(&zc);
        let groups = (0..128)
            .map(|c| {
                let v = &trees[c][layer + 1];
                let n = v.len() / 2;
                Group {
                    left: v[..n].to_vec(),
                    right: v[n..].to_vec(),
                    scale: eqc[c],
                }
            })
            .collect::<Vec<_>>();
        let (x, rx, pairs) = if layer == 0 {
            (
                None,
                vec![],
                groups
                    .iter()
                    .map(|g| [g.left[0], g.right[0]])
                    .collect::<Vec<_>>(),
            )
        } else {
            let (p, r, v) = prove_eq(t, groups, &zx, layer)?;
            (Some(p), r, v)
        };
        let g = Group {
            left: pairs.iter().map(|p| p[0]).collect(),
            right: pairs.iter().map(|p| p[1]).collect(),
            scale: F::ONE,
        };
        let (c, rc, pair) = prove_eq(t, vec![g], &zc, layer)?;
        let pair = pair[0];
        tagged(t, 0x32, &pair, json!({"layer":layer}))?;
        let mu = draw(
            t,
            "gkr.child_line_challenge",
            json!({"layer":layer,"coordinate":"mu"}),
        )?;
        zx = rx;
        zx.push(mu);
        zc = rc;
        layers.push(Layer { x, c, pair });
    }
    let eqx = eq_table(&zx);
    let eqc = eq_table(&zc);
    let a = (0..32768).map(|r| eqx[r] * (powers[r] + F::ONE)).collect();
    let b = (0..32768)
        .into_par_iter()
        .map(|r| (0..128).fold(F::ZERO, |v, c| if table.bit(c, r) { v + eqc[c] } else { v }))
        .collect();
    let (presum, mut point, mu) = prove_presum(t, a, b)?;
    point.extend_from_slice(&zc);
    Ok((
        Proof {
            folds,
            layers,
            presum,
        },
        point,
        mu,
    ))
}
pub fn verify(t: &mut Transcript, p: &Proof, opening: &Bitified) -> io::Result<(Vec<F>, F)> {
    if p.folds.len() != 128 || p.layers.len() != 15 || p.folds.contains(&u128::MAX) {
        return Err(io::Error::other("forest shape or fold range"));
    }
    let comb = FixedBasePow::new(F::from(2u64), 8);
    let roots = common::column_images(&comb, &p.folds);
    tagged(t, 0x30, &roots, json!({}))?;
    let mut zc = draw_point(t, "gkr.tree_batching_point", 7)?;
    let mut claim = evaluate(&roots, &zc);
    let mut zx = vec![];
    for (layer, l) in p.layers.iter().enumerate() {
        let rx = if layer == 0 {
            if l.x.is_some() || l.c.sum != claim {
                return Err(io::Error::other("root layer claim"));
            }
            vec![]
        } else {
            let x =
                l.x.as_ref()
                    .ok_or_else(|| io::Error::other("missing forest phase A"))?;
            if x.sum != claim {
                return Err(io::Error::other("forest input claim"));
            }
            let (r, v) = verify_eq(t, x, &zx, layer)?;
            if v != eq(&r, &zx) * l.c.sum {
                return Err(io::Error::other("forest phase A terminal"));
            }
            r
        };
        let (rc, v) = verify_eq(t, &l.c, &zc, layer)?;
        if v != eq(&rc, &zc) * l.pair[0] * l.pair[1] {
            return Err(io::Error::other("forest closing pair"));
        }
        tagged(t, 0x32, &l.pair, json!({"layer":layer}))?;
        let mu = draw(
            t,
            "gkr.child_line_challenge",
            json!({"layer":layer,"coordinate":"mu"}),
        )?;
        claim = l.pair[0] + mu * (l.pair[0] + l.pair[1]);
        zx = rx;
        zx.push(mu);
        zc = rc;
    }
    if p.presum.sum != claim + F::ONE {
        return Err(io::Error::other("presum input claim"));
    }
    let (mut point, v) = verify_presum(t, &p.presum)?;
    let weights = eq_table(&zx);
    let powers = common::row_images(&comb, &opening.rows);
    let a = weights
        .iter()
        .zip(powers)
        .map(|(&e, p)| e * (p + F::ONE))
        .collect::<Vec<_>>();
    let divisor = evaluate(&a, &point)
        .inv()
        .ok_or_else(|| io::Error::other("zero presum divisor"))?;
    point.extend(zc);
    Ok((point, v * divisor))
}
