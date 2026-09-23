//! Nonce search and validation shared by PCS grinding rounds.
//!
//! A positive difficulty requires that many leading zero bits in
//! `BLAKE3(POW_HASH_TAG || seed || nonce.to_le_bytes())`. At zero difficulty,
//! only nonce zero is accepted. Transcript framing belongs to each calling round.

const POW_HASH_TAG: &[u8] = b"bitz-pcs-pow-v1";

/// Returns the first valid nonce in ascending order.
// todo: parallel pow? use potentially spongefish?
pub(crate) fn find(seed: &[u8; 16], bits: u32) -> u64 {
    if bits == 0 {
        return 0;
    }
    let mut nonce = 0u64;
    loop {
        if valid(seed, nonce, bits) {
            return nonce;
        }
        nonce = nonce.checked_add(1).expect("proof-of-work nonce exhausted");
    }
}

/// Checks the hash difficulty, or the canonical zero nonce at zero difficulty.
pub(crate) fn valid(seed: &[u8; 16], nonce: u64, bits: u32) -> bool {
    if bits == 0 {
        return nonce == 0;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(POW_HASH_TAG);
    hasher.update(seed);
    hasher.update(&nonce.to_le_bytes());
    let digest = hasher.finalize();
    leading_zero_bits(digest.as_bytes()) >= bits
}

fn leading_zero_bits(bytes: &[u8]) -> u32 {
    let mut total = 0;
    for byte in bytes {
        let zeros = byte.leading_zeros();
        total += zeros;
        if zeros != 8 {
            break;
        }
    }
    total
}
