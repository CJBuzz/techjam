# TechJam demonstration-overlap audit

## Verdict

**PASS — no supplied COCO-val2017 or DALL·E Advanced demonstration image entered training, model selection, or calibration for `diverse_initialized_40k_calibrated.pt`.**

The audit was run on 2026-08-31 with `scripts/audit_demo_overlap.py`. Every supplied image and every influential dataset row was independently decoded to RGB. SHA-256 was computed over width, height, and raw RGB bytes. An overlap, decode failure, missing manifest file, or mismatch with the persisted manifest hash was defined as a failure.

## Recovered training identity

- Split manifest: `data/mixed_wildfake_66k/split_manifest.csv`
- Manifest SHA-256: `6214c16fc37b7652d2b8a5a059506b7257b3bb6eaae005bbe649c3c40c6bf028`
- Train: 47,994 images
- Model selection: 9,988 images
- Calibration: 2,000 images
- Total influential rows rehashed: 59,982
- Reserved test: 6,000 images, excluded from this audit because it was not extracted or scored while producing the checkpoint

Training sources were CIFAKE, SID labels 0/1, WildFake ImageNet real, and WildFake DDIM, DDPM, BigGAN, and StyleGAN fake. Model selection added held-out ADM fake plus ImageNet real. Calibration contained only the persisted CIFAKE/SID calibration rows. Manifest provenance contained zero occurrences of `COCO`, `val2017`, `DALL-E`, `DALLE`, or `Advanced`.

## Original demonstration archive

- Archive: `wildfake-001.zip`
- Archive SHA-256: `8a871ca61e4e5aaef7baa6e7638842a6263347838bc2f4c36362b28e1a85d9c8`
- Images decoded: 13,841
- DALL·E Advanced: 8,843
- COCO val2017: 4,998
- Unique decoded-pixel hashes: 8,717
- Decode failures: 0
- Dataset stored-hash mismatches after recomputation: 0
- Exact decoded-pixel overlaps: **0**
- Local report: `artifacts/mixed_wildfake_66k/demo_overlap_audit.json`
- Report SHA-256: `b090d6f2ca9b6042d3cc65b7dd367bd06d748955fcd169439f2b8f6c28d0c70d`

The lower unique-hash count reflects duplicate decoded pixels within the supplied archive; every one of the 13,841 members was still decoded and checked.

## Transformed demonstration archive

The exact transformed archive used for validation was audited separately as an additional safeguard.

- Archive: `wildfake_robust-002.zip`
- Archive SHA-256: `4b8e5f24e684018b62eb64ad33c478afe98260bee8f8a28b721947f29f631d43`
- Images decoded: 13,841
- Unique decoded-pixel hashes: 13,803
- Decode failures: 0
- Dataset stored-hash mismatches after recomputation: 0
- Exact decoded-pixel overlaps: **0**
- Local report: `artifacts/mixed_wildfake_66k/demo_robust_overlap_audit.json`
- Report SHA-256: `bb56f0ac77323584508f45bcd76920a7047cdc6954c2a3a69dc4eb4452bf26c3`

## Reproduction

Run the audit once for each supplied archive:

```powershell
python scripts/audit_demo_overlap.py `
  --data-root data/mixed_wildfake_66k `
  --split-manifest data/mixed_wildfake_66k/split_manifest.csv `
  --demo-zip <archive.zip> `
  --output <report.json> `
  --workers 8
```
