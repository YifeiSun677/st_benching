# Linear baselines on her2st — runbook

Three baselines, all leave-one-patient-out (A–H), same 13,620 spots, 833 genes,
folds and scorer as the deep-model ports:

| Baseline | What it predicts | Scored on |
|---|---|---|
| `train_mean` | each gene's mean over the 7 training patients | all 36 sections (PCC is NaN by definition → use MAE) |
| `region_mean` | mean of the spot's pathologist label over training patients' annotated sections | the 8 annotated sections (A1 B1 C1 D1 E1 F1 G2 H1) |
| `ridge` | closed-form ridge on frozen features; α by inner leave-one-patient-out | all 36 sections |

Nothing is ever fitted, scaled, PCA'd or selected on the held-out patient.

## Where things run

| Step | Machine |
|---|---|
| 0. check | RunPod (CPU) |
| 1. align features (re-uses the caches the other ports wrote) | RunPod (CPU) |
| 1b. extract a missing cache (only `uni_native224`, if absent) | RunPod (GPU) |
| 2. run baselines | RunPod (CPU, minutes) |
| 3. score | RunPod (CPU) |
| 4. figures | Mac, after `rsync` of `$RIDGE_OUT` (small CSV/JSON/npz) |

## 0. Check the volume (read-only)

```bash
cd /workspace/st_benching
python -m ridge.check_workspace \
  --preds TRIPLEX=/workspace/triplex_results/triplex_lopo_833 \
          STFlow=/workspace/runs/stflow_lopo_833_hest112_e90 \
          Path2Space=<path2space run dir>
```

Must be PASS before going on: canonical spot index (13,620), raw counts,
`shared truth == cpm_panel_ln(raw counts)`. Feature sources may be MISSING; only
the ones you plan to use matter. `--preds` reports which target space each
model was trained/scored in.

## 1. Align feature caches to the canonical spot order

```bash
python -m ridge.features --sources all        # skips sources whose cache is absent
```

Writes `$RIDGE_CACHE/features_<source>.npz`. Fails loudly if any spot is missing.

If `uni_native224` is missing and you want it (GPU, a few minutes):

```bash
python stflow/build_features.py --patch_mode native224
python -m ridge.features --sources uni_native224
```

## 2. Run

```bash
# Baseline 1
python -m ridge.run_lopo --model train_mean
python -m ridge.run_lopo --model region_mean

# Baseline 3: general UNI ridge, shared target, local and with linear context
python -m ridge.run_lopo --model ridge --source uni_hest112 --context local
python -m ridge.run_lopo --model ridge --source uni_hest112 --context full

# Frozen twins, each in its deep model's own target space (fair pairing)
python -m ridge.run_lopo --model ridge --source uni_hest112 --context local --target native   # vs STFlow
python -m ridge.run_lopo --model ridge --source cigar       --context local --target native   # vs TRIPLEX
python -m ridge.run_lopo --model ridge --source cigar       --context full  --target native   # vs TRIPLEX (context)
python -m ridge.run_lopo --model ridge --source resnet50    --context local --target native   # vs DeepPT
python -m ridge.run_lopo --model ridge --source ctranspath  --context local --target native   # vs Path2Space
```

Each run prints the chosen α per fold; `fold.json` keeps the full inner-CV
curve. A warning `HIGH EDGE`/`LOW EDGE` means the grid should be extended.

## 3. Score

```bash
python -m ridge.score --tag <tag>                          # all sections
python -m ridge.score --tag <tag> --sections annotated     # same 8 sections as region_mean
python -m ridge.score --run-dir <any port's run dir> --sections annotated --out-dir <dir>
```

Same definition as `triplex/score.py` (verified to 1e-9 on the TRIPLEX run),
plus MAE and the number of NaN genes.

## Fair pairings

| Deep model | Frozen twin source | Target (`--target native`) | Patch |
|---|---|---|---|
| STFlow | `uni_hest112` | `raw_ln` (old run) — use `--target cp10k_panel_ln` for the normalised run | 112 µm → 224 px |
| TRIPLEX | `cigar` | `cpm_panel_ln` | shared 224 px |
| DeepPT | `resnet50` | `cp10k_full_log10` | DeepPT crops |
| Path2Space | `ctranspath` | `mediannorm_panel_ln` | Macenko 224 px |
| BLEEP, ST-Net, HisToGene, Hist2ST | no frozen twin (fine-tuned or trained from scratch) | compare with `uni_native224` / `uni_hest112` in their target | — |

HisToGene's truth is `cp10k_panel_log10`; its PCC equals `cp10k_panel_ln`'s.

## Tests (run anywhere)

```bash
python -m pytest ridge/tests -q
```

Checks: ridge == scikit-learn Ridge on standardised features; target transforms
reproduce TRIPLEX / Path2Space / STFlow / DeepPT / HisToGene stored truth;
scorer reproduces the TRIPLEX headline; end-to-end run on the real counts with
synthetic features.
