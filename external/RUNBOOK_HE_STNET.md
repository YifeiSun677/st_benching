# Runbook — ST-Net transfer: her2st → He et al. 2020

Apply the 8 her2st LOPO ST-Net checkpoints (`densenet121_224/top_833`, epoch 25) to the He et al. 2020
breast cohort (the ST-Net paper data), with the same input, target and scoring as the Visium arm.

**Cohort.** Mendeley `10.17632/29ntw7sh4r` (v5): 23 patients, 68 sections, 0.76 GB.
Same legacy ST platform as her2st (100 µm spots, 200 µm pitch), so there are no pseudo-spots and
no platform gap. Only the cohort and the scanner change. The subtypes are HER2_luminal (5 patients),
HER2_non_luminal (5), Luminal_A (4), Luminal_B (5) and TNBC (4). **HER2+** (the first two, 10 patients)
is the same disease as her2st and is the primary transfer group. Luminal and TNBC are out-of-disease.

**Files per He section** (checked on BC23287_C1). Names come from `metadata.csv`, because the BT/BC
prefixes are not consistent:

| file | content |
|---|---|
| `HE_BT23287_C1.jpg` | full-res H&E, 9311×9856, ≈ 291 px per 200 µm = **0.687 µm/px** |
| `spots_BT23287_C1.csv.gz` | index `XxY` (array col × row), `X`,`Y` = pixel column,row; under-tissue spots only (256) |
| `BC23287_C1_stdata.tsv.gz` | spots × **ENSG** counts, all ~1000 array spots, genes with 0 counts dropped |
| `BC23287_C1_Coords.tsv.gz` | tumour / non labels per spot (CR line endings, misaligned header) |

**Code** (all in `external/`; `common.py` and `run_stnet.py` are reused unchanged):

| step | file |
|---|---|
| paths, readers, output writer | `he_common.py` |
| 1 download | `download_he.py` |
| 1.1 verify | `verify_he_raw.py` |
| 2 resample + her2st layout | `export_he_like.py` (needs `ext/calib/her2st_scale.json` from `calib_her2st_scale.py`) |
| 3 her2st held-out + round-trip | `run_stnet.py --sections` (existing Visium driver, no Visium sections) |
| 4 predict He | `run_stnet_he.py` |
| 5 score | `score_he.py` (reuses `score_external.score_one`) |

**Pod layout**

```
/workspace/ext/he/raw/                    downloads, metadata.csv, MANIFEST.tsv, dataset.json
/workspace/ext/he/her2st_like/data/       ST-imgs/<patient>/<SEC>/HE_<SEC>.jpg  (resampled)
                                          ST-spotfiles/<SEC>_selection.tsv, ST-cnts/<SEC>.tsv.gz
/workspace/ext/he/calib/                  raw_summary.tsv, he_scale.tsv, counts_<SEC>.npz,
                                          panel_coverage.tsv, qc/<SEC>.jpg
/workspace/runs/he_stnet/fold0<k>_<P>/preds/<SEC>.npz
/workspace/results/he/
```
`<SEC>` = patient_replicate, e.g. `BC23287_C1`.

---

## 0. Sync and preflight (pod)

```bash
cd /workspace/st_benching && git pull
pip install -q opencv-python-headless pillow scipy h5py openslide-python openslide-bin
```

`openslide` is needed by ST-Net itself (steps 3–4). `openslide-bin` ships the C library as a wheel;
if pip can't install it, use `apt-get install -y openslide-tools` instead.

```bash
cd /workspace && R=ST-Net/output/densenet121_224/top_833; for P in A B C D E F G H; do ls $R/${P}_checkpoints/epoch_25.pt $R/${P}_gene.log $R/${P}_25.npz >/dev/null || echo "MISSING fold $P"; done; ls panels/panel_train.txt ext/calib/her2st_scale.json; python -c 'import openslide; print("openslide ok")'
```

Nothing should print "MISSING". If `her2st_scale.json` is missing, run `python external/calib_her2st_scale.py`
(Visium-arm Stage 2.1). It must be the same file the Visium arm used, so both arms share one scale.

## 1. Download

```bash
cd /workspace/st_benching && python external/download_he.py
```

- The script lists all 273 files via the Mendeley API, downloads the 272 needed plus metadata, and checks sha256.
- The endpoint sometimes returns a JSON error body with HTTP 200. Those downloads are retried automatically.
- **Check:** the last line reads `273/273 files verified, 0.76 GB`. If it exits 1, rerun it (good files are kept).
- For HER2+ only (about 1/3 of the data): `--subtypes HER2_luminal HER2_non_luminal`.

### 1.1 Verify raw

```bash
python external/verify_he_raw.py
```

- **Expect** `PASS`, with um/px close to 0.687 for every section. BC23803 (TNBC) has larger JPEGs; check that its um/px matches the rest.
- Per section, `spots_with_counts == n_spots`, `ensg True`, and `tumor_labelled` ≈ `n_spots`.
- If it fails: `xy_diff_pct > 3` or `rmse_frac_pitch > 0.10` means a broken spot file; `spots_inside_image False` means the wrong image. Report the section and don't `--force` blindly.

## 2. Resample to her2st scale and export

```bash
nohup python external/export_he_like.py --jobs 8 > /workspace/ext/he/calib/export.log 2>&1 &
```

CPU-only. `--jobs N` exports N sections in parallel (about 1–1.5 GB RAM each). Pick N at most `nproc`, and at most free RAM / 1.5 GB.
The summary tables are written once at the end, so never launch several copies of the script instead.

Per section, this:

1. fits He µm/px from its own spot grid;
2. computes `f = µm/px_He / µm/px_her2st`;
3. resizes the H&E (INTER_AREA if f < 1, LANCZOS4 if f > 1) and writes JPEG Q95;
4. scales the spot pixels by f;
5. keeps spots that have counts and > 0 UMI;
6. maps ENSG → symbol with the **HGNC** table, choosing the name her2st uses: current symbol, else a previous
   symbol, else an alias, and only names present in her2st (for example `AES`, now `TLE5`). One table covers all
   68 sections (`calib/ensg_to_symbol.tsv`). ST-Net's own `ensembl.tsv` on the pod is an empty stub, so ST-Net
   was trained on her2st symbols directly.

**Checks**

- One JSON line per section. `f` should be nearly constant across sections, `n_dropped` small, `n_edge_spots` ~0 (these crops get black padding).
- The first line `ENSG -> symbol … panel genes reachable N/833` and the last line `panel genes present … N/833`
  should both read **824/833** or very close. The 9 missing genes are `IGHA1 IGHG3 IGHG4 IGHM IGKC IGLC2 IGLC3 IGLC7 TRAC`,
  which are absent from He's annotation. They are scored as unmeasured (dropped), not as zeros.
- **Look at 3–4 `calib/qc/<SEC>.jpg`.** Circles must sit on tissue (red = tumour, blue = non, green = unlabelled), and the black box shows one 224-px window. A local test on BC23287_C1 lined up cleanly.
- `calib/he_scale.tsv` is the bookkeeping table. Copy it into the results.

## 3. her2st held-out predictions in the same run tree (paired baseline + round-trip)

```bash
python external/run_stnet.py --sections --out /workspace/runs/he_stnet
```

- `--sections` with no values means no Visium sections. This runs only the her2st held-out path: ST-Net's own `Spatial` dataset, compared with the stored `<P>_25.npz`.
- **Check:** `ROUNDTRIP PASS`. This also proves the checkpoints, the normalisation stats parsed from `<P>_gene.log`, and the gene permutation all load correctly.
- It upserts the ST-Net rows of `ext/roundtrip.tsv` with the same values as before.

## 4. Predict He

Smoke test first (one fold, one section):

```bash
python external/run_stnet_he.py --folds B --sections BC23287_C1 --out /workspace/runs/he_stnet_smoke
```

**Check the printed lines:**

- `gene universe …` shows the first id. `features mapped X by ENSG + Y by symbol` shows how He columns entered the her2st universe. Expect X large if `gene.pkl` holds ENSG ids, or Y large if it holds symbols; either is fine. If both are small, stop.
- The PCC should be finite and positive.

Then the full run (8 models loaded at once; each section is cropped once):

```bash
python external/run_stnet_he.py --out /workspace/runs/he_stnet 2>&1 | tee /workspace/runs/he_stnet/predict.log
```

**Check:** each of the 8 fold dirs holds 68 He npz plus that fold's her2st held-out npz:

```bash
for d in /workspace/runs/he_stnet/fold0*; do echo "$d $(ls $d/preds | wc -l)"; done
```

`rm -rf /workspace/runs/he_stnet_smoke` afterwards so the scorer (which globs `he_*`) doesn't pick it up.

## 5. Score

```bash
python external/score_he.py --models stnet
```

It writes `/workspace/results/he/{per_section,per_patient_fold,paired_delta,summary}.csv` and `per_gene_pcc.csv.gz`.

- Metrics come from `score_external.score_one`: same gene filter (panel ∧ measured ∧ SD > 0) and same SSE baseline as the Visium arm.
- `folds = 8fold`: patients are averaged within each fold, then mean/min/max over the 8 folds.
- `folds = ens`: the mean of the 8 fold predictions per spot (an extra row; not in the paired delta).
- `delta_pcc_mean_*`: each He patient minus the **same fold's** her2st held-out patient. This is the transfer drop.
- Groups: each subtype, `HER2+`, `He_all`, and `her2st_heldout` (the reference row).

Bring back to the laptop:

```bash
mkdir -p results/he_stnet && scp -r <pod>:/workspace/results/he/* results/he_stnet/ && scp <pod>:/workspace/ext/he/calib/{he_scale.tsv,raw_summary.tsv} results/he_stnet/
```

---

## Things to keep in mind when reading the numbers

- **Same platform, new site.** He et al. and her2st come from overlapping groups (KTH/Lund) but are different
  studies. I could not verify here that no patient appears in both. If you know of any overlap, drop those
  patients before reporting.
- **No stain normalisation**, the same as the Visium arm. Stain shift is part of what the transfer measures.
- **Target universe.** Z sums only over genes in ST-Net's her2st `gene.pkl`. He features outside it are
  ignored, which is the same convention as Visium.
- **Tumour labels** are kept in `ST-spotfiles/<SEC>_selection.tsv` (`tumor` column) for a later tumour/stroma split. They are not used in scoring yet.
- The her2st-like tree (`ext/he/her2st_like/data`) is ready for the other ports (HisToGene, Hist2ST, …):
  counts are symbols and images are at her2st scale. Section ids are longer than her2st's 2-char ids,
  so each port's driver has to take the section list explicitly.

---

# Part B — BLEEP transfer (same He export, steps 1–2 are shared)

Driver: `run_bleep_he.py`. It reuses `run_bleep.py` (the Visium driver) and the port in `bleep/` unchanged:

- **Checkpoints:** `/workspace/runs/bleep_lopo_e10/<P>/last.pt`.
- **Reference bank:** the fold's 7 training patients from `/workspace/her2st_cache`, encoded by the expression encoder.
- **Prediction:** the mean of the top-50 retrieved reference spots.
- **Truth:** BLEEP's own target (panel counts → CPM → log1p), computed from `ext/he/her2st_like/data/ST-cnts`.

The only override is `her2st_dataset.find_image`, which otherwise assumes `ST-imgs/<SEC[0]>/<SEC>/`.

## B.0 Preflight

```bash
cd /workspace && for P in A B C D E F G H; do ls runs/bleep_lopo_e10/$P/last.pt runs/bleep_lopo_e10/$P/preds.npz >/dev/null || echo "MISSING $P"; done; ls her2st_cache/index.json her2st_cache/patches.npy
```

```bash
cd /workspace/st_benching && python -c "import sys; sys.path[:0]=['external','bleep']; import run_bleep; print('bleep ok')"
```

If the import fails, install from `bleep/requirements_bleep.txt`.

## B.1 her2st held-out predictions in the same run tree (paired baseline + round-trip)

```bash
python external/run_bleep.py --sections --out /workspace/runs/he_bleep 2>&1 | tee /workspace/runs/he_bleep_b1.log
```

**Check:** `ROUNDTRIP PASS`. Retrieval is discrete, so small |ΔPCC| (≤ 1e-3) is expected; see `run_bleep.py --rt_tol`.

## B.2 Smoke test, then the full run

```bash
python external/run_bleep_he.py --folds B --sections BC23287_C1 --out /workspace/runs/he_bleep_smoke
```

Expect one line with `316`-ish spots, `0 edge-padded`, and a finite, positive PCC. Then:

```bash
rm -rf /workspace/runs/he_bleep_smoke && python external/run_bleep_he.py --out /workspace/runs/he_bleep 2>&1 | tee /workspace/runs/he_bleep/predict.log
```

This loads 8 CLIP models and 8 reference banks at once; each He image is loaded once. If GPU memory is tight
(for example while ST-Net is still running), use `--folds A,B,C,D` and then `--folds E,F,G,H`. The outputs land in the same tree.

## B.3 Score both models together

```bash
python external/score_he.py
```

With no `--models`, it scores every `runs/he_*` directory into one set of tables. Running `--models bleep` on its own
**overwrites** `results/he/*` with only BLEEP, so score both models together (or pass a different `--out`).

**BLEEP-specific caveat.** The CPM denominator is the panel total. The 9 panel genes He never measured
(IGHA1 IGHG3 IGHG4 IGHM IGKC IGLC2 IGLC3 IGLC7 TRAC) are high in immune-rich her2st spots but zero in He,
so He spot totals are slightly smaller and every other gene's CPM slightly larger. The effect is a mild
per-spot rescaling, applied by BLEEP's own transform. It is kept, not corrected, so the target stays BLEEP's.

---

# Part C — HisToGene transfer

Driver: `run_histogene_he.py`. It reuses `run_histogene.py` (the Visium driver) and the **main-table port**:
the `histogene/` package modules `cache.py`, `config.py`, `her2st.py` and `dataset.py`, with weights from
`runs/histogene_lopo_833_ckpt`. The "Re-organize files" commit had deleted those four modules; they were
restored unchanged from `92ded7e`. The newer `dataset_fast.py` / `train_lopo.py` port is a different run.

- **Input:** a whole He section per forward pass, 112-px crops of the her2st-scale JPEG.
- **Positions:** He array `x, y`, which are 200-µm units like her2st, so no `x_int` is needed.
- **Truth:** log10(CP10K + 1) over the panel (`histogene.her2st.expression`).

## C.0 Preflight

```bash
cd /workspace && for k in 0 1 2 3 4 5 6 7; do ls -d runs/histogene_lopo_833_ckpt/fold0${k}_* >/dev/null || echo "MISSING fold $k"; done; ls runs/histogene_lopo_833_ckpt/fold00_A/; ls -d /workspace/HisToGene /workspace/cache/htg_patch112
```

```bash
cd /workspace/st_benching && python -c "import sys; sys.path.insert(0,'external'); import run_histogene; print('histogene ok')"
```

Each fold dir must hold `last.ckpt` and `preds/`.

## C.1 her2st held-out + round-trip

```bash
python external/run_histogene.py --sections --out /workspace/runs/he_histogene 2>&1 | tee /workspace/runs/he_histogene_c1.log
```

**Check:** `ROUNDTRIP PASS`. Same weights, so expect an exact match (`--rt_tol 1e-4`).

## C.2 Smoke test, then the full run

```bash
python external/run_histogene_he.py --folds B --sections BC23287_C1 --out /workspace/runs/he_histogene_smoke
```

```bash
rm -rf /workspace/runs/he_histogene_smoke && python external/run_histogene_he.py --out /workspace/runs/he_histogene 2>&1 | tee /workspace/runs/he_histogene/predict.log
```

All 68 sections are cropped once (about 1.5 GB RAM), then each fold loads one model and predicts every section.

---

# Part D — Path2Space transfer

Driver: `run_path2space_he.py`. It reuses `run_path2space.py` (the Visium driver: Macenko pool helpers)
and the `path2space/` port:

- **Features:** frozen CTransPath on 224-px tiles, per-tile Macenko.
- **Ensemble:** 7 ik × 7 il MLPs per fold from `/workspace/p2s_ckpt/path2space_lopo_833_ckpt/fold_P`.
- **Truth:** the `lognorm` target (panel library × section median, log1p).
- **Footing:** RAW, with no smoothing.

## D.0 Preflight

```bash
ls /workspace/p2s_ckpt/path2space_lopo_833_ckpt/fold_A/ /workspace/p2s_weights/ctranspath.pth /workspace/p2s_upstream/ge_model/path2space >/dev/null && echo ok; pip install -q spams-bin opencv-python-headless
```

```bash
cd /workspace/st_benching && python -c "import sys; sys.path.insert(0,'external'); import run_path2space; print('p2s ok')"
```

## D.1 her2st held-out + round-trip

```bash
python external/run_path2space.py --sections --out /workspace/runs/he_path2space 2>&1 | tee /workspace/runs/he_path2space_d1.log
```

**Check:** `ROUNDTRIP PASS`. It reads the her2st feature cache, so no Macenko runs here.

## D.2 He features (CPU Macenko, then GPU CTransPath; cached and resumable)

```bash
python external/run_path2space_he.py --features_only --workers 48 2>&1 | tee /workspace/runs/he_path2space_features.log
```

Macenko is the slow part: about 27k tiles in total. `--workers` can go well above the default 16 on this 128-core pod.
Tiles go to `ext/he/features/path2space/tiles_<SEC>.npz`, then become `<SEC>.npz` features.
If the run stops, rerunning continues where it left off.

## D.3 Predict

```bash
python external/run_path2space_he.py --out /workspace/runs/he_path2space 2>&1 | tee /workspace/runs/he_path2space/predict.log
```

Features are cached, so this takes seconds per fold (only MLPs).

---

# Scoring all models together

```bash
python external/score_he.py
```

This scores every `runs/he_*` directory into `results/he/`, with ST-Net, BLEEP, HisToGene and Path2Space side by side.
Each model is paired only with its **own** her2st held-out (its C.1/D.1-style step), so every delta stays within one model.

**Target spaces differ by model:** ST-Net log((1+c)/(n+Z)), BLEEP CPM-log1p, HisToGene log10(CP10K+1),
Path2Space median-lognorm. PCC and the paired delta are compared across models; absolute SSE is not.
