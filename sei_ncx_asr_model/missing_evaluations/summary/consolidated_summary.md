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

## Morpheme F1 (word-level token F1 proxy)

| Configuration | Language | Token F1 before | Token F1 after | Δ |
|---|---|---|---|---|
| legacy B4 (B1 preds + recon) | sei | 0.0099 | 0.0276 | +0.0177 |
| legacy B4 (B1 preds + recon) | ncx | 0.0655 | 0.1562 | +0.0906 |
| joint_adapter_on_sei | sei | 0.4408 | 0.4491 | +0.0083 |
| joint_adapter_on_ncx | ncx | 0.4964 | 0.5167 | +0.0202 |
| seed_ncx_seed13_baseline | ncx | 0.5785 | 0.5785 | +0.0000 |
| seed_ncx_seed42_baseline | ncx | 0.5820 | 0.5820 | +0.0000 |
| seed_sei_seed13_baseline | sei | 0.5396 | 0.5396 | +0.0000 |
| seed_sei_seed42_baseline | sei | 0.5148 | 0.5148 | +0.0000 |

## Notes for the paper

- Monolingual WERs are reported as mean ± std across all available seeds (including the original reference adapter if re-evaluated via this script).
- Joint (unbalanced) uses plain concatenation, matching the archived XLS-R methodology.
- Joint (upsampled) repeats ncx entries to match sei volume. Comparing the two joint rows disambiguates capacity interference from data dilution.
- τ=1.0 is the classic B4 always-on; τ<1.0 is B4.5 (uncertainty-triggered). Best τ is typically 0.5–0.7 in these runs.
- Morpheme F1 is computed via `eval_morpheme_f1.py`. The metric is word-level token F1 and word-boundary F1 — an ad-hoc proxy, since no morphological analyser exists for sei/ncx.