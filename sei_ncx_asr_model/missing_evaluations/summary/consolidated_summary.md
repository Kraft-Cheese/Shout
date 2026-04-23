# Shout — consolidated results after missing-evaluations run

## WER grid

| Configuration | sei test | ncx test |
|---|---|---|
| Zero-shot Whisper-small (B1) | 1.3512 | 1.7771 |
| sei adapter (monolingual) | 0.5561 ± 0.0156 (n=2) | — |
| ncx adapter (monolingual) | — | 0.5458 ± 0.0031 (n=2) |
| joint adapter (unbalanced) | 0.6326 | 0.6686 |

## τ-sweep (WER after reconstruction)

| Configuration | τ=0.0 | τ=0.3 | τ=0.5 | τ=0.7 | τ=1.0 |
|---|---|---|---|---|---|
| **joint_adapter_on_sei** | 0.6326 | 0.6289 | 0.6201 | 0.6149 | 0.6190 |
| **joint_adapter_on_ncx** | 0.6686 | 0.6635 | 0.6556 | 0.6476 | 0.6527 |

## Morpheme F1 (token + boundary proxy)

| Configuration | Language | Token F1 before | Token F1 after | Token Δ | Boundary F1 before | Boundary F1 after | Boundary Δ |
|---|---|---|---|---|---|---|---|
| legacy B4 (B1 preds + recon) | sei | 0.0099 | 0.0276 | +0.0177 | 0.1402 | 0.1463 | +0.0061 |
| legacy B4 (B1 preds + recon) | ncx | 0.0655 | 0.1562 | +0.0906 | 0.1956 | 0.2242 | +0.0286 |
| sei_ref_on_ncx | ncx | 0.0405 | 0.1204 | +0.0799 | 0.1768 | 0.2279 | +0.0510 |
| ncx_ref_on_sei | sei | 0.0056 | 0.0417 | +0.0361 | 0.1343 | 0.1530 | +0.0187 |
| joint_adapter_on_sei | sei | 0.4408 | 0.4491 | +0.0083 | 0.4101 | 0.4340 | +0.0239 |
| joint_upsampled_on_ncx | ncx | 0.4997 | 0.5112 | +0.0116 | 0.5578 | 0.5655 | +0.0077 |
| seed_ncx_seed42_baseline | ncx | 0.5820 | 0.5820 | +0.0000 | 0.6257 | 0.6257 | +0.0000 |
| ncx_seed7_on_ncx | ncx | 0.5401 | 0.5349 | -0.0053 | 0.5916 | 0.5997 | +0.0081 |
| ncx_ref_on_ncx | ncx | 0.5433 | 0.5421 | -0.0012 | 0.5779 | 0.5868 | +0.0089 |
| sei_seed7_on_sei | sei | 0.4857 | 0.4802 | -0.0056 | 0.4401 | 0.4495 | +0.0094 |
| sei_ref_on_sei | sei | 0.4721 | 0.4683 | -0.0037 | 0.4167 | 0.4354 | +0.0187 |
| seed_sei_seed13_baseline | sei | 0.5396 | 0.5396 | +0.0000 | 0.4787 | 0.4787 | +0.0000 |
| joint_upsampled_on_sei | sei | 0.4520 | 0.4560 | +0.0039 | 0.4286 | 0.4424 | +0.0138 |
| seed_sei_seed42_baseline | sei | 0.5148 | 0.5148 | +0.0000 | 0.4613 | 0.4613 | +0.0000 |
| joint_adapter_on_ncx | ncx | 0.4964 | 0.5167 | +0.0202 | 0.5404 | 0.5741 | +0.0336 |
| seed_ncx_seed13_baseline | ncx | 0.5785 | 0.5785 | +0.0000 | 0.6124 | 0.6124 | +0.0000 |

## Notes for the paper

- Monolingual WERs are reported as mean ± std across all available seeds (including the original reference adapter if re-evaluated via this script).
- Joint (unbalanced) uses plain concatenation, matching the archived XLS-R methodology.
- Joint (upsampled) repeats ncx entries to match sei volume. Comparing the two joint rows disambiguates capacity interference from data dilution.
- τ=1.0 is the classic B4 always-on; τ<1.0 is B4.5 (uncertainty-triggered). Best τ is typically 0.5–0.7 in these runs.
- Morpheme F1 is computed via `eval_morpheme_f1.py`. The metric is word-level token F1 and word-boundary F1 — an ad-hoc proxy, since no morphological analyser exists for sei/ncx.