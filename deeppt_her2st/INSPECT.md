# Read this before you trust any number

The Zenodo archive blocks automated download, so `deeppt_models.py` and the
defaults in `03_run_lopo.py` are a **faithful-in-spirit reconstruction**, not a
transplant. Before the real run, unzip the archive and replace my guesses with
the published values. Everything below is a 20-minute job and it is the
difference between "I ran DeepPT" and "I ran something DeepPT-shaped".

```bash
cd /workspace/DeepPT_original
sed -n '1,200p' 12AE/1main_AE.py
sed -n '1,200p' 13DeepPT_train/1main_train.py
ls 13DeepPT_train/          # there may be a model.py / utils.py alongside
```

## Checklist — 12AE/1main_AE.py

| What to find | Where it goes |
|---|---|
| encoder/decoder layer widths and depth (is 2048→512 one layer or several?) | `deeppt_models.AE` |
| activation (ReLU? tanh? none on the code layer?) | `deeppt_models.AE` |
| whether the AE input is standardised, and if so how | `run_fold`, the `mu/sd` block |
| optimiser, LR, weight decay, batch size, epochs | `--ae-lr --wd --batch --ae-epochs` |
| loss (plain MSE, or MSE + a sparsity/KL term?) | `train_ae` |

## Checklist — 13DeepPT_train/1main_train.py

| What to find | Where it goes |
|---|---|
| predictor depth/width, dropout, activation | `deeppt_models.Predictor` |
| **whether the target is z-scored per gene** — this is the trap | see below |
| optimiser, LR, scheduler, weight decay, batch size | `--mlp-lr --wd --batch` |
| loss function (MSE? Huber? correlation-based?) | `run_fold` |
| early-stopping criterion and patience | `--patience` |
| whether tile predictions are averaged to slide level | **delete this** — one tile per spot |

### The z-score trap

If `1main_train.py` z-scores expression per gene before training, predictions
come back in z-space. PCC will still look completely normal (it is invariant to
affine rescaling), but MSE, R² and gain-over-baseline will be nonsense, and the
numbers will silently not match your ST-Net / HisToGene tables. If the original
does z-score, either (a) invert it before saving in `03_run_lopo.py`, or
(b) don't z-score and say so in methods. Do not leave it ambiguous.

## Two deviations from the paper you must state in methods

1. **Spot-level, not slide-level.** DeepPT components (i) and the tile→slide
   averaging step are removed; one tile per spot, target is that spot's
   expression. Same adaptation DeepHis2Exp and Wang et al. 2025 make.
2. **Patch size 224, not 512@20×.** Chosen for comparability with your ST-Net
   window-224 runs.

## The stain-normalisation decision

Your benchmark-wide `--colornorm raw` rule was justified on the grounds that
`reinhard` was DeepHis2Exp's own addition. **That reasoning does not apply
here** — colour normalisation is part of DeepPT's published pipeline,
explicitly to suppress staining heterogeneity and batch effects. So for DeepPT,
`raw` is a deviation *from* the original, not a return to it.

Recommended: `raw` as the primary result (13-model consistency), plus one fold
re-run with `--colornorm macenko` as a sensitivity check. Only step 1 repeats
for that fold's sections, so it costs ~5 minutes.
