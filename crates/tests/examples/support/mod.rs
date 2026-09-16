use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
};

use field::F128;

pub fn write_binary(
    path: impl AsRef<Path>,
    write: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>,
) -> io::Result<()> {
    let mut output = BufWriter::new(File::create(path)?);
    write(&mut output)?;
    output.flush()
}

pub fn write_witness(mut output: impl Write, packed: &[F128]) -> io::Result<()> {
    for element in packed {
        output.write_all(&element.lo.to_le_bytes())?;
        output.write_all(&element.hi.to_le_bytes())?;
    }
    Ok(())
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn witness_format_preserves_little_endian_words_and_element_order() {
        let packed = [
            F128::new(0x0706050403020100, 0x0f0e0d0c0b0a0908),
            F128::new(0x1716151413121110, 0x1f1e1d1c1b1a1918),
        ];
        let mut bytes = Vec::new();
        write_witness(&mut bytes, &packed).unwrap();
        assert_eq!(bytes, (0..32).collect::<Vec<u8>>());
    }
}
