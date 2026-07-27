# Reproducibility levels

## Level 1: artifact reproduction

Run `python reproduce.py`. This validates the compact Origin worksheets for
Figures 2-3 and redraws Figures 4-5 and S1-S3 from the direct figure-source
CSVs. Figures 1-3 were authored in Origin; their final images and complete
plotted worksheet values are retained directly.

Run `python reproduce.py --verify-only` to verify the archived source and
reference files against `SHA256SUMS.txt`.

## Level 2: method-contract verification

Run `python -m pytest`. The tests check weight bounds, convex blending and the
absence of a test-label argument from reliability-feature construction.

## Level 3: raw-data model refitting

This level requires the four source datasets and their redistribution terms.
They are not committed here. Before claiming full raw-data reproducibility,
the authors must:

1. deposit each permitted dataset or give stable accession/download details;
2. document checksums and preprocessing versions;
3. run the four preprocessors and four unified entrypoints documented in the
   root README;
4. record software versions and verify fresh outputs against the retained
   figure-source values within declared tolerances.

The SSE study source is identified by the versioned ChemRxiv DOI
`10.26434/chemrxiv.15005550/v1`. Because the analysed workbook contains
author-curated revisions, turnkey raw-data reproduction additionally requires
the revised workbook checksum and a transformation record relative to the
deposited source.
