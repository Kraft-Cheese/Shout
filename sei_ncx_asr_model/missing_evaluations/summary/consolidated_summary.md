# Shout — consolidated results after missing-evaluations run

## WER grid

| Configuration | sei test | ncx test |
|---|---|---|
| Zero-shot Whisper-small (B1) | 1.3512 | 1.7771 |
| sei adapter (monolingual) | 0.5798 ± 0.0292 (n=4) | 1.4515 |
| ncx adapter (monolingual) | 1.0305 | 0.5764 ± 0.0356 (n=4) |
| joint adapter (unbalanced) | 0.6326 | 0.6686 |
| joint adapter (upsampled) | 0.6160 | 0.6816 |

## τ-sweep (WER after reconstruction)

| Configuration | τ=0.0 | τ=0.3 | τ=0.5 | τ=0.7 | τ=1.0 |
|---|---|---|---|---|---|
| **joint_adapter_on_sei** | 0.6326 | 0.6289 | 0.6201 | 0.6149 | 0.6190 |
| **joint_adapter_on_ncx** | 0.6686 | 0.6635 | 0.6556 | 0.6476 | 0.6527 |
| **sei_ref_on_sei** | 0.6090 | 0.6046 | 0.6021 | 0.6024 | 0.6035 |
| **ncx_ref_on_ncx** | 0.6027 | 0.5991 | 0.5926 | 0.5970 | 0.6143 |
| **sei_ref_on_ncx** | 1.4515 | 1.4399 | 1.4110 | 1.3907 | 1.3763 |
| **ncx_ref_on_sei** | 1.0305 | 1.0294 | 1.0243 | 1.0154 | 1.0059 |
| **joint_upsampled_on_sei** | 0.6160 | 0.6149 | 0.6068 | 0.6046 | 0.6068 |
| **joint_upsampled_on_ncx** | 0.6816 | 0.6795 | 0.6657 | 0.6657 | 0.6744 |
| **sei_seed7_on_sei** | 0.5980 | 0.5929 | 0.5892 | 0.5855 | 0.5925 |
| **ncx_seed7_on_ncx** | 0.6114 | 0.6107 | 0.6071 | 0.6064 | 0.6208 |

## Morpheme F1 (word-level token F1 proxy)

| Configuration | Language | Token F1 before | Token F1 after | Δ |
|---|---|---|---|---|
| legacy B4 (B1 preds + recon) | sei | 0.0099 | 0.0276 | +0.0177 |
| legacy B4 (B1 preds + recon) | ncx | 0.0655 | 0.1562 | +0.0906 |
| sei_ref_on_sei | sei | 0.4721 | 0.4683 | -0.0037 |
| ncx_ref_on_ncx | ncx | 0.5433 | 0.5421 | -0.0012 |
| sei_ref_on_ncx | ncx | 0.0405 | 0.1204 | +0.0799 |
| ncx_ref_on_sei | sei | 0.0056 | 0.0417 | +0.0361 |
| joint_upsampled_on_sei | sei | 0.4520 | 0.4560 | +0.0039 |
| joint_upsampled_on_ncx | ncx | 0.4997 | 0.5112 | +0.0116 |
| sei_seed7_on_sei | sei | 0.4857 | 0.4802 | -0.0056 |
| ncx_seed7_on_ncx | ncx | 0.5401 | 0.5349 | -0.0053 |
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