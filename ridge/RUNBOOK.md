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
          Path2Space=/workspace/st_benching/results/path2space_lopo_833
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

Replace the example tag with your run's folder name in `$RIDGE_OUT` (e.g. `ls /workspace/ridge_results`):

```bash
python -m ridge.score --tag ridge_uni_hest112_local_cpm_panel_ln                       # all sections
python -m ridge.score --tag ridge_uni_hest112_local_cpm_panel_ln --sections annotated  # same 8 sections as region_mean
python -m ridge.score --run-dir /workspace/triplex_results/triplex_lopo_833 --sections annotated --out-dir /workspace/ridge_results/triplex_annotated
```

Same definition as `triplex/score.py` (verified to 1e-9 on the TRIPLEX run),
plus MAE and the number of NaN genes.

## Native-scale UNI ridge (one baseline recipe, scored on each model's own scale)

Every deep model is scored on the scale it was trained on, so the general UNI ridge
(and region mean) are re-run on each of those scales. Already run: `cpm_panel_ln`
(TRIPLEX, BLEEP) and `raw_ln` (STFlow). The rest (RunPod, CPU, minutes each):

```bash
# needs panels/stnet_denominator.csv (made from ST-Net's stored truth by ridge.stnet_denominator)
for t in cp10k_panel_log10 cp10k_full_log10 mediannorm_panel_ln stnet_lognorm; do
  for c in local full; do
    python -m ridge.run_lopo --model ridge --source uni_hest112 --context $c --target $t
  done
done
```

| Target | Scored against |
|---|---|
| `cp10k_panel_log10` | HisToGene |
| `cp10k_full_log10` | DeepPT |
| `mediannorm_panel_ln` | Path2Space, and Hist2ST (its log10 version; identical PCC) |
| `stnet_lognorm` | ST-Net |

Then copy the 8 run dirs (`ridge_uni_hest112_{local,full}_<target>`) to the Mac's
`results_4_models/ridge/`.

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

## Training-size experiment (ridge vs TRIPLEX as the number of training patients grows)

Question: does the TRIPLEX − ridge gap widen with more training patients? Every
comparison is within one held-out test patient.

| Piece | Code | Design |
|---|---|---|
| subset list (both pipelines read it) | `tsize/manifest.py` → `tsize/manifest.json` | every size-k subset of the other 7 patients, k ∈ {1,2,4,6,7}: 71 per test patient, 568 per config; TRIPLEX: 2 per k ∈ {2,4} for test patients B, C, A, E, drawn (seed 20261117) from the ridge list |
| ridge | `ridge/run_tsize.py` | 9 configs: UNI local × 5 scales, cigar local + full, resnet50 local, ctranspath local; α fixed to the k = 7 choice per test patient (`k7`) and the per-spot-scaled version (`k7_per_spot` = α·n_train/n_train(k=7)) |
| TRIPLEX | `triplex/run_tsize.py` (+ `train_patients=` in `train.py`) | 16 runs; hyperparameters checked against `triplex_lopo_833_e20/fold_*/run.json` before training; k = 7 = that run |
| tables | `tsize/aggregate.py` | tidy table + `by_k.csv`, `paired_gap*.csv`, `k1_matrix*` |

The α values come from the existing k = 7 runs and are already written into
`tsize/manifest.json`, so the pod only reads that file. Rebuild it only if a k = 7
run changes: `python -m tsize.manifest --k7-dir ../results_4_models/ridge`.

### 0. Get the code onto the pod

Commit and push on the Mac (including `tsize/manifest.json`), then on the pod:

```bash
cd /workspace/st_benching && git pull && git log -1 --oneline
```

### 1. Tests on the pod (CPU, ~5 min)

```bash
cd /workspace/st_benching && python -m pytest tsize/tests -q
```

The 9 `test_k7_fixed_alpha_reproduces_stored_run_real[...]` cases must PASS, not
skip: they refit every config at k = 7 with the stored α on the real features and
compare against `/workspace/ridge_results/<tag>` (preds to 1e-5, per-gene PCC to
1e-6). A skip means a feature file or k = 7 run is missing. (The TRIPLEX dry-run
test skips on the pod; step 3a does the same check against the real index.)

### 2. Ridge (CPU, about 1 h for all 9 configs; resumable)

Run it in tmux so it survives an SSH disconnect. If tmux is missing on the pod:

```bash
apt-get update && apt-get install -y tmux
```

Start a detached session (the log is also kept on the volume; the shell stays
open after the job ends so you can read the last lines):

```bash
tmux new -d -s tsize_ridge 'cd /workspace/st_benching && python -u -m ridge.run_tsize 2>&1 | tee -a /workspace/tsize_ridge.log; exec bash'
```

Do not run ridge at the same time as TRIPLEX training. The pod's CPU quota
(`cat /sys/fs/cgroup/cpu.max`, about 10 cores) is far below `nproc` (128), so the
two jobs fight over it: ridge slowed to about 2.5 s per fit and TRIPLEX to about
50 s per epoch. Run them one after the other. With the pod to itself, ridge is
faster as two jobs that split the configs between them (about 0.2–0.4 s per fit),
each capped with `OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5` and
given its own `--configs` list. If you script a wait for another job, use
`pgrep -f "[t]riplex.run_tsize"`; without the brackets, pgrep matches the waiting
shell's own command line, so the wait never ends.

Watch it (detach again with Ctrl-b then d; the job keeps running):

```bash
tmux attach -t tsize_ridge
```

Progress without attaching:

```bash
cd /workspace/st_benching && python -m ridge.run_tsize --status
```

tmux sessions die if the pod restarts. In that case start the same session again;
finished fits are skipped.

Measured cost per fit: ~0.3 s at d = 1024 (UNI), ~1.3 s at d = 2048 (resnet50), ~1 s
for cigar full (its global PCA is refitted per subset). Roughly: UNI 5 × ~4 min,
cigar local ~1 min, cigar full ~10 min, resnet50 ~15 min, ctranspath ~3 min.
If it is interrupted, run the same command again; finished fits are skipped.
Output: `/workspace/ridge_results/tsize/<tag>/<test>/<subset_id>.csv.gz`
(9 × 568 = 5,112 files).

### 3. TRIPLEX (GPU, 16 runs, about 1–1.5 h; resumable)

On a fresh pod, first re-do run.sh's setup steps only (deps + weights link), not its training:

```bash
cd /workspace/st_benching && export TRIPLEX_REPO=/workspace/TRIPLEX CIGAR_CKPT=/workspace/weights/cigar/tenpercent_resnet18.ckpt
```
```bash
pip install --no-input -r triplex/requirements-triplex.txt && mkdir -p weights/cigar && ln -sfn "$CIGAR_CKPT" weights/cigar/tenpercent_resnet18.ckpt
```

3a. Dry run: prints each run's split, spot count, share of the k = 7 data and
optimizer steps; refuses to continue if any hyperparameter differs from the e20 run.

```bash
python -m triplex.run_tsize --dry-run
```

3b. Train (k = 7 folds took 9–10 min for 20 epochs; k = 2 ≈ 3 min, k = 4 ≈ 6 min):

```bash
nohup python -m triplex.run_tsize > /workspace/tsize_triplex.log 2>&1 &
```
```bash
tail -f /workspace/tsize_triplex.log
```

If it is interrupted, run the same command again: finished runs (final.pt + preds)
are skipped, and a half-finished run continues from its resume.pt (every 10 epochs).
Output: `/workspace/triplex_results/triplex_tsize_<subset_id>/fold_<test>/`.
Each `run.json` records `train_patients`, `n_train_spots`, `steps_per_epoch` and
`total_steps`. With the epoch count fixed at 20, k = 2 gets about 2/7 of the steps
of k = 7, so check those numbers when reading the gap.

### 4. Aggregate (CPU, ~2 min) and pull to the Mac

```bash
cd /workspace/st_benching && python -m tsize.aggregate
```

This writes `/workspace/ridge_results/tsize_summary/` and prints the TRIPLEX − ridge(cigar)
gap per test patient and k. On the Mac (from `~/Documents/science`; use the pod's
current IP and port):

```bash
scp -r -P <port> -i ~/.ssh/id_ed25519 'root@<ip>:/workspace/ridge_results/tsize_summary' results_4_models/
```
```bash
scp -r -P <port> -i ~/.ssh/id_ed25519 'root@<ip>:/workspace/ridge_results/tsize' results_4_models/ridge_tsize
```
```bash
scp -r -P <port> -i ~/.ssh/id_ed25519 'root@<ip>:/workspace/triplex_results/triplex_tsize_*' results_4_models/triplex/
```

Check the copy, then optionally re-aggregate locally (same output):

```bash
find results_4_models/ridge_tsize -name '*.csv.gz' | wc -l; ls -d results_4_models/triplex/triplex_tsize_* | wc -l
```
```bash
cd st_benching && python -m tsize.aggregate --ridge-dir ../results_4_models/ridge_tsize --triplex-dir ../results_4_models/triplex --out ../results_4_models/tsize_summary
```

Expected: 5112 and 16.

## External cohorts: ridge on He et al. and TCGA (Fig 6)

The her2st reference baseline (UNI, local) applied to the two transfer cohorts. No new fitting choices:
`ridge/external_fit.py` refits each of the 8 folds on its 7 training patients with the alpha that fold already
chose in `ridge_uni_hest112_local_cp10k_panel_log10`, on target `cp10k_panel_ln` (the He / TCGA scale; ln vs
log10 only rescales the output). Outputs use the same contracts as the deep models, so the existing scorers run
unchanged. Model name: `ridge_uni`.

### 0. Code and tests (pod, CPU)

```bash
cd /workspace/st_benching && git pull && python -m pytest ridge/tests/test_external_fit.py -q
```

Then check the refit against the stored her2st run (needs `/workspace/ridge_cache/features_uni_hest112.npz` and
`/workspace/ridge_results/ridge_uni_hest112_local_cp10k_panel_log10`). Must print PASS:

```bash
python -m ridge.external_fit --verify
```

### 1. He et al. (CPU, minutes)

The features are STFlow's cached He UNI embeddings. Count them (expect 68):

```bash
ls /workspace/ext/he/features/stflow/*.npz | wc -l
```

Only if fewer than 68 (GPU, once):

```bash
python external/run_stflow_he.py --features_only
```

Run and score. Score into its OWN folder: `score_he.py` overwrites its output files with only the models it is given.

```bash
python -m ridge.run_he --verify
```
```bash
python external/score_he.py --models ridge_uni --out /workspace/results/he_ridge_uni
```

### 2. TCGA (GPU for UNI, then CPU; resumable)

UNI was never cached for TCGA, so this embeds every window once (same crop as STFlow's TCGA run) and keeps the
embeddings with `--cache` so a rerun skips the GPU. Run in tmux:

```bash
tmux new -d -s tcga_ridge 'cd /workspace/st_benching && python -u -m ridge.run_tcga --verify --cache /workspace/ext/tcga/features/uni_hest112 2>&1 | tee -a /workspace/tcga_ridge.log; exec bash'
```

Try `--limit 6` first if you want a quick timing. When it finishes, pseudo-bulk and score (both write per-model
file names, so other models' files are untouched):

```bash
python tcga/pseudobulk.py --runs /workspace/runs/tcga_ridge_uni --out /workspace/results/tcga
```
```bash
python tcga/score_tcga.py --pb /workspace/results/tcga/pseudobulk_ridge_uni.tsv.gz
```

The Mac has TCGA scores only for ST-Net, DeepPT and Hist2ST. If the other five models were never scored, the same
two commands per model (`runs/tcga_<model>` -> `pseudobulk_<model>` -> `scores_<model>`):

```bash
for m in bleep histogene path2space stflow triplex; do python tcga/pseudobulk.py --runs /workspace/runs/tcga_$m --out /workspace/results/tcga && python tcga/score_tcga.py --pb /workspace/results/tcga/pseudobulk_$m.tsv.gz; done
```

### 3. Copy to the Mac (from `~/Documents/science`; use the pod's current IP and port)

```bash
scp -r -P <port> -i ~/.ssh/id_ed25519 'root@<ip>:/workspace/results/he_ridge_uni' ext_results/he_stnet/results/
```
```bash
scp -r -P <port> -i ~/.ssh/id_ed25519 'root@<ip>:/workspace/results/tcga/*' tcga_results/tcga/
```

The Mac's `ext_results/he_stnet/results/he_cp10k/per_gene_pcc.csv.gz` has all 104 He sections only for ST-Net and
Hist2ST (one section for the other six). Copy the full He result folder for all 8 models too -- the folder the
`he_cp10k` scoring was written to on the pod (check with `ls -d /workspace/results/he*`):

```bash
scp -r -P <port> -i ~/.ssh/id_ed25519 'root@<ip>:/workspace/results/he_cp10k' ext_results/he_stnet/results/he_cp10k_full
```
