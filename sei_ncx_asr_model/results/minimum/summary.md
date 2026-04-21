# Shout — minimum experiment suite results

## Results grid (WER)

| Configuration | sei test | ncx test |
|---|---|---|
| sei adapter | — | — |
| ncx adapter | — | — |
| joint adapter | 0.6326 | 0.6686 |

## τ-sweep (B4/B4.5 unified)

| Config | τ=0.0 | τ=0.3 | τ=0.5 | τ=0.7 | τ=1.0 |
|---|---|---|---|---|---|
| | (passthrough) | | | | (B4 always-on) |
| **joint_adapter_on_sei** (sei) | 0.6326 | 0.6289 | 0.6201 | 0.6149 | 0.6190 |
| **joint_adapter_on_ncx** (ncx) | 0.6686 | 0.6635 | 0.6556 | 0.6476 | 0.6527 |

## Morpheme F1 (at τ=1.0, i.e. B4 always-on)

| Config | Token F1 before | Token F1 after | Δ |
|---|---|---|---|