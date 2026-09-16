Reference export from f2z-pcs `bitz-parity-logging-transcript-wrapper`, HEAD
`5aac69a5e21dddbc60ef95bdec957485b88568dd` plus the working changes for logical
logging, plain-UDR and the Spartan-only exporter. Generated with
`u32_mul_transcript --variant plain-udr --through-spartan --fixture canonical`.
Only diagnostic span arrays were removed. Values, wire chunks, roles, sequences
and draw counts are unchanged. The fixture is test-only and never drives proving.
Regenerate a fresh export with the documented parity workflow before updating it.
