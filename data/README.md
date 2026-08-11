# Data provenance

This repository does not silently bundle third-party participant-level data. The reproduction
pipeline distinguishes downloaded source data from small, cited aggregate values.

## HAIID (primary analysis data)

The paper reuses the public **Human-AI Interactions Dataset (HAIID)** released by Vodrahalli et al.
The complete dataset contains 35,670 interactions from 1,125 participants across five tasks. The
main policy evaluation uses the Art, Sarcasm, Cities, and Census tasks under the
`perceived_accuracy=80` condition.

Run:

```bash
python scripts/download_data.py
```

The downloader retrieves two files from the upstream repository at the pinned commit
`24881cc7586180a9c9742a7dd838aea97d008235` and verifies their SHA-256 checksums before use.

| File | SHA-256 |
|---|---|
| `haiid_dataset.csv` | `be9223b6bf34f996cdace9b1c0d43876df0e480bcb9322e6a7f774de0f2f0eed` |
| `haiid_dataset_description.csv` | `2d5fe97cf0af1ae67bff402eae073f6bc1a92442a648af73d8470ef8c691560d` |

- Upstream repository: <https://github.com/kailas-v/human-ai-interactions>
- Upstream license: MIT, copyright Kailas Vodrahalli (2022)
- Dataset paper: Vodrahalli et al., *Do Humans Trust Advice More if it Comes from AI?*
  (AIES 2022)

Downloaded files are placed in `data/raw/HAIID/` and excluded from Git.

## GRACE (bounded complementary evidence)

`data/external/grace_verbalized_results.csv` contains six aggregate rows transcribed from Table 1
of the published GRACE paper. They document the bounded comparison between conventional ECE and
the human-grounded CalScore. GRACE is **not** an external validation of the HAIID display policies.

- Paper: Sung et al., *GRACE: A Granular Benchmark for Evaluating Model Calibration against Human
  Calibration*, ACL 2025, <https://doi.org/10.18653/v1/2025.acl-long.962>
- Upstream code/data: <https://github.com/Pinafore/advcalibration>

The upstream GRACE repository did not expose a license in the audited snapshot, so this repository
does not redistribute its raw benchmark files.

## Prospective feasibility pilot

The accepted paper reports a small 20-participant pilot as descriptive feasibility evidence. Its
participant-level data are not present in the research workspace and are therefore not claimed as
reproducible by this package. The repository reproduces only the offline HAIID analyses and the
published aggregate GRACE comparison.
