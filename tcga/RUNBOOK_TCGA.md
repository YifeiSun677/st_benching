# Runbook — TCGA-BRCA: her2st models → whole-slide pseudo-bulk vs bulk RNA-seq

The third generalisation test. The her2st LOPO checkpoints of all nine models (no training, no
fine-tuning) predict every tissue spot of whole TCGA slides. Spot predictions are averaged into a
per-patient pseudo-bulk and compared with that patient's bulk RNA-seq.

## Design (pre-registered — fixed before any prediction is scored)

| item | choice |
|---|---|
| cohort | TCGA-BRCA, 100 patients, fixed-seed random draw per stratum (`select_cohort.py`, seed 20260929) |
| strata | **HER2+** (clinical, any PAM50) 40 · **LumA** 20 · **LumB** 20 · **Basal** 20 (clinical HER2−, PAM50) |
| eligible | female · no neoadjuvant therapy · primary-tumour STAR counts · ≥1 frozen slide (TS/BS) from the same sample as the RNA · ≥1 DX slide · determinable clinical HER2 (FISH over IHC; IHC+/FISH− excluded) |
| slides | **all** TS/BS and **all** DX of each patient, **whole slide** (every tissue spot) |
| conditions | **Frozen** = all TS+BS spots pooled (primary: same block and preparation as the RNA; frozen, like her2st) · **DX** = all DX spots pooled (FFPE, a different block) · **TS vs BS** agreement (patients with both) as a reproducibility ceiling |
| pseudo-bulk | each model's spot predictions → its own inverse transform → linear scale → mean over the slide's spots (sum and n are stored, so slide- and patient-level means are exact) · 8 folds averaged, fold spread kept |
| bulk target | the 833-gene panel, **panel CPM** (counts / panel total × 1e6), as the spot-level training targets · full-transcriptome TPM secondary |
| primary statistic | within each stratum: per-gene Spearman across patients → **median over the gene set** (all / HVG / SVG / marker) |
| significance | patient-label permutation within the stratum (1000×) |
| model comparison | paired bootstrap over patients of the median-r difference |
| analyses | within HER2+ (primary; same disease as training) · within LumA, LumB, Basal (out-of-disease). No pooled analysis |
| not done | stain normalisation, tumour-region filtering, spot-level artefact QC (all as in the Visium and He arms) |
| replacement | a slide that fails (`no_mpp`, `bad_mpp`, `no_tissue`) → if the patient loses all frozen or all DX slides, take the next **reserve** patient of that stratum in rank order (`cohort.csv`) |

---

## A. Mac (done — CPU only)

Environment: `~/Documents/science/.venv_tcga` (system site-packages + `openslide-python openslide-bin`).
All large inputs are streamed and deleted; only the upload package stays.

### A.1 Cohort — `python tcga/select_cohort.py`

GDC files API + cBioPortal (`brca_tcga` for HER2 IHC/FISH, sex, neoadjuvant; `brca_tcga_pan_can_atlas_2018`
for PAM50). Raw API responses snapshot in `results/tcga_cohort/raw/` (`--offline` reproduces the draw).
Outputs in `results/tcga_cohort/`: `eligibility.csv` (all 1,101 patients, first failing rule),
`cohort.csv` (eligible patients, stratum, rank, role main/reserve), `slides.csv`, GDC manifests, `summary.json`.

### A.2 Slides → her2st-like virtual sections — `python tcga/prep_slides.py --jobs 12`

Per slide, streamed (download → md5 → process → delete `.svs`):

1. **mpp** from the slide (`openslide.mpp-x/y`); missing or implausible → status `no_mpp` / `bad_mpp`.
2. **Tissue mask** at 4 µm/px: HSV saturation > max(Otsu, 15); pen removed (green hue; black =
   dark **and** unsaturated — dense frozen tissue is dark but saturated and stays tissue); open/close;
   components < 0.02 mm² dropped. **No blue-ink rule**: a first version flagged hue 95–125 as blue ink
   and removed bluish tumour nests (TCGA-BH-A18H DX1: 8.8 mm², BH-A0HA 5.3 mm²); it was dropped and
   every slide re-run. Blue ink, when present, stays in the mask (usually thin lines on the glass
   around tissue, rarely ≥ 50 % of a 100-µm spot) — a known limitation.
3. **Grid**: legacy ST, 200 µm pitch, 100 µm spots, origin at the tissue bbox corner; a spot is kept
   when ≥ 50 % of its disk is tissue.
4. **Windows** (every kept spot in exactly one): tissue clusters (pieces ≥ 2 grid steps apart are
   separate) → each split into ≤ 33 × 35 spots (her2st array), balanced → windows with < 20 spots join
   the nearest window if the merged extent stays ≤ 48 (HisToGene/Hist2ST need array coords < 64).
5. **Export** in the her2st layout, image resampled to her2st **0.6882 µm/px**
   (`ext_results/calib/her2st_scale.json`, the same reference as the Visium and He arms), JPEG Q95.
   Pixels > 224 px (Chebyshev) from every kept spot are white; no model reads > 112 px from a spot.
   `ST-cnts` are **placeholders** (all 1) so the ports' readers work; they are never a target.
6. QC thumbnail per slide (`qc/<patient>_<token>.jpg`): mask outline, pen (magenta), spots coloured
   by window, window boxes.

Section name: `<patient>_<slide token>_w<NN>`, e.g. `TCGA-A2-A04U_TSA_w01`.

Download failures (GDC read timeouts, DNS drops) are not recorded as done: rerunning the same command
retries only those slides.

### A.2b Ink filter — `python tcga/ink_filter.py`

The mask's black rule (V < 60 **and** S < 100) misses near-black pixels, whose HSV saturation is
numerically unstable (V ≈ 10 reads S ≈ 160–180). Surgical-margin ink on tissue (TCGA-D8-A13Y DX1)
and a scanner black corner (TCGA-EW-A1PH DX1) therefore got spots. This step re-checks every exported
spot on the full-resolution JPEG: ink = (V < 60 and S < 100) **or** V < 30; a spot with ≥ 50 % ink in its
100-µm disk is dropped (mirrors the 50 % tissue rule). Images untouched; spot files and placeholder
counts rewritten; empty windows removed; `meta/ink_filter.tsv` logs it. On the first 625 windows:
69 spots dropped, 2 windows emptied (both all-ink).
Checked by eye on one DX and one TS slide: mask, grid, windows, and spot ↔ pixel alignment
(224-px crop at her2st scale shows tumour nuclei at her2st size).

### A.3 Bulk RNA — `python tcga/prep_rna.py`

STAR counts (GENCODE v36), unstranded. ENSG → her2st symbol with `he_common.ensg_to_symbol` (the He
rule) plus one fix (`resolve_ambiguous`): when the first previous symbol already belongs to another
gene, use the other unowned previous symbol (VARS1 → `VARS`, QARS1 → `QARS`, RMP24 → `C18orf21`).
**833/833** panel genes measured. Outputs in `meta/rna/`: `rna_panel_counts.tsv`, `rna_panel_cpm.tsv`
(patients × 833), full `rna_all_counts.tsv.gz` / `rna_all_tpm.tsv.gz`, mapping and coverage tables.

> **He arm check (pod):** the same first-hit rule ran there. Look up ENSG00000204394 (VARS1),
> ENSG00000172053 (QARS1) and ENSG00000141428 in `ext/he/calib/ensg_to_symbol.tsv`: if any maps to
> `VARS2`/`QARS2`-style names, the He panel counts for those genes were mislabelled.

### A.3b Result (2026-09-29)

All **253/253** slides `ok` (8 needed a second run for download timeouts); no patient replaced.
One slide was a **duplicate scan**: GDC holds two files for TCGA-A8-A06U-01A-01-TS1 (scanners SS1436
and SS12035, same tissue). Both were processed under the same section names, so only one could be on
disk. Rule (now in `select_cohort.py`): one scan per (sample, token), lowest file_id — keeps
`4571fd46…`, which is the one on disk; the other's record moved to `meta/slides_excluded/`.
Final: **252 slides, 1,429 windows, 450,442 spots**, 27.1 GB JPEG. Every patient has frozen and DX spots.

| spots per patient | min | median | max |
|---|---|---|---|
| Frozen (TS+BS) | 20 (TCGA-E2-A1LK) | 926 | 5,848 |
| DX | 165 (TCGA-AC-A3W5) | 2,889 | 10,098 |

Windows: median 281 spots (her2st sections 177–712); max extent 48 × 48; **16 windows have < 20 spots**
(isolated tissue pieces that cannot be merged; smallest = 1 spot) — include them in the B.3 smoke test.
Sensitivity analysis candidate: drop patients with very few frozen spots (e.g. < 100).

### A.4 Pack — `python tcga/pack_upload.py --max-gb 8`

| batch | patients | sections | spots | GB |
|---|---|---|---|---|
| 1 | 16 | 358 | 134,677 | 8.0 |
| 2 | 23 | 382 | 130,171 | 8.0 |
| 3 | 35 | 463 | 133,169 | 8.0 |
| 4 | 26 | 228 | 52,658 | 3.2 |

`~/Documents/science/tcga_upload/`:

| file | content |
|---|---|
| `batch_NN.tar` | sections of whole patients, ≤ 8 GB each, uncompressed |
| `meta.tar.gz` | `meta/` (params.json, slides.tsv, windows.tsv with `batch`, batches.tsv, rna/, cohort files) + `qc/` |
| `meta/`, `qc/` | the same, loose, for browsing on the Mac |

---

## B. Pod

Start with **ST-Net**: it predicts each 224-px patch on its own (no neighbours, no section context,
no counts), so it tests the core path — TCGA image → spot prediction → pseudo-bulk → bulk RNA —
without any window effects. The other eight follow the same B.2–B.5 pattern.

### B.0 Code

```bash
cd /workspace/st_benching && git pull
python -c "import sys; sys.path.insert(0,'external'); import tcga_common, run_stnet_tcga; print('ok')"
```

New files: `external/tcga_common.py`, `external/run_stnet_tcga.py`, `tcga/pseudobulk.py`, `tcga/score_tcga.py`.

### B.1 Upload and extract

From the Mac (resumable; rerun the same command after a drop):

```bash
rsync -avP --partial --inplace -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" ~/Documents/science/tcga_upload/meta.tar.gz ~/Documents/science/tcga_upload/batch_0*.tar root@<IP>:/workspace/ext/tcga/
```

On the pod (each tar is deleted once extracted, so the peak is ~ upload + 8 GB):

```bash
cd /workspace/ext/tcga && mkdir -p her2st_like && tar -xzf meta.tar.gz && for b in batch_0*.tar; do tar -xf "$b" -C her2st_like && rm "$b"; done
ls her2st_like/data/ST-imgs | wc -l          # 100 patients (fewer if you upload batch by batch)
ls her2st_like/data/ST-spotfiles | wc -l     # 1429 (all batches)
```

### B.2 Smoke test — ST-Net (~5 min)

```bash
cd /workspace/st_benching
python external/run_stnet_tcga.py --he-check BC23287_C1
```

**Expect** 8 lines `max |pred_tcga_path - pred_he_stored| = …e-0x` and `HE-CHECK PASS`. This proves the
TCGA code path is the He code path. FAIL → stop and report the numbers.

```bash
python external/run_stnet_tcga.py --out /workspace/runs/tcga_stnet_smoke --sections \
  TCGA-A2-A04U_TSA_w01 TCGA-E2-A1LL_DX1_w02 TCGA-A8-A06U_BS1_w01 \
  $(awk -F'\t' 'NR>1 && $5==1 {print $1; exit}' /workspace/ext/tcga/meta/windows.tsv)
```

(the last one is a 1-spot window, TCGA-A2-A04Q_TSB_w03; `$5` = `n_spots` in windows.tsv. The four
smoke sections are in **batches 3 and 4** — upload those first if you go batch by batch.)

**Expect** per section: `N spots, median across-spot SD of raw pred X` with X > 0 (predictions vary
across spots; the 1-spot window prints `nan`), then `DONE … s per 1000 spots`. Files:
`/workspace/runs/tcga_stnet_smoke/fold0{0..7}_{A..H}/agg/<SEC>.npz`.
Smoke-test checks 2 (placeholder invariance) is automatic for ST-Net: the TCGA driver never opens `ST-cnts`.

### B.3 Full run — ST-Net

```bash
rm -rf /workspace/runs/tcga_stnet_smoke
nohup python external/run_stnet_tcga.py --out /workspace/runs/tcga_stnet > /workspace/runs/tcga_stnet.log 2>&1 &
tail -f /workspace/runs/tcga_stnet.log
```

450k spots × 8 folds; use the smoke test's `s per 1000 spots` for the time (densenet121 on one GPU:
order of an hour). Resumable: rerun the same command after an interruption. Batch-by-batch upload:
run after each batch is extracted (`--batch N` restricts to one batch); already-done sections are skipped.

Output: 1,429 × 8 small files in `agg/` (+ full spot matrices for the rank-1 patient of each stratum in `spots/`).

### B.4 Pseudo-bulk and scores — ST-Net

```bash
python tcga/pseudobulk.py --runs /workspace/runs/tcga_stnet --out /workspace/results/tcga
python tcga/score_tcga.py --pb /workspace/results/tcga/pseudobulk_stnet.tsv.gz
```

**Expect** from `pseudobulk.py`: `1429/1429 windows; missing 0` and `100 patients, {frozen: 100, DX: 100,
TS: 94, BS: 47}`. A `WARNING incomplete patients` line means some sections were not predicted yet.

`score_tcga.py` prints, per condition (frozen, DX) × stratum × gene set (all, marker):
`median_r` (primary), `ci_lo/ci_hi`, `frac_pos`, `p_perm`, fold range; then `TSvsBS` rows (prediction
reproducibility between the two faces of the block, and each face vs RNA). Full tables:
`scores_stnet.tsv`, `per_gene_stnet.tsv.gz`.

Pipeline check done on the Mac with synthetic models: a model = RNA + noise scored median r ≈ 0.8, p = 1/(B+1);
a random model ≈ 0.

### B.5 BLEEP and DeepPT (patch-level, like ST-Net)

| model | driver | output raw → lin | he-check PASS rule |
|---|---|---|---|
| BLEEP | `external/run_bleep_tcga.py` | mean log1p(panel CPM) of top-50 retrieved her2st spots → `expm1` | ≥ 99 % spots identical and corr > 0.999 per fold (retrieval is discrete) |
| DeepPT | `external/run_deeppt_tcga.py` | log10(CP10K+1) → `max(10**raw − 1, 0)` | max \|diff\| < 1e-2 and corr > 0.9999 (fp16 encoder) |

Environment after a pod restart: `pip install -r bleep/requirements_bleep.txt` (BLEEP: timm, …).
DeepPT features are **not cached** (3.7 GB): each section is encoded once and fed to the 8 fold heads.
Same sequence as ST-Net: `--he-check BC23287_C1` → smoke on the four B.2 sections → full run →
`pseudobulk.py --runs /workspace/runs/tcga_<model>` → `score_tcga.py --pb …/pseudobulk_<model>.tsv.gz`.

### B.6 Spatial models: HisToGene, Hist2ST, TRIPLEX, STFlow, Path2Space

| model | driver | raw → lin | notes |
|---|---|---|---|
| HisToGene | `run_histogene_tcga.py` | log10(CP10K+1) → `max(10**x−1,0)` | positions remapped (below) |
| Hist2ST | `run_hist2st_tcga.py` | log10(x/lib·median+1) → `max(10**x−1,0)` | positions remapped; k capped at n−1 (upstream `calcADJ` raises IndexError on blocks with ≤ k spots); self-loop on isolated spots (He fix) |
| TRIPLEX | `run_triplex_tcga.py` | log1p(panel CPM) → `max(expm1,0)` | APEG grid scanned per fold as He; neighbours on the whole window; position + global branch per block |
| STFlow | `run_stflow_tcga.py` | log1p(panel CP10K) → `max(expm1,0)` | one seeded `predict()` per block; row-index placeholder labels; `--placeholder-check` |
| Path2Space | `run_path2space_tcga.py` | lognorm log1p → `max(expm1,0)` | Macenko in a bounded `spawn` pool (no tiles on disk); CPU-bound (~10-core quota on this pod) |

**Position remapping** (`tcga_common.her2st_blocks`): her2st training sections use array x ∈ [2, 32],
y ∈ [2, 34] only; TCGA windows start at 1 and reach 48, so learned position embeddings would be indexed
outside the trained rows. Each window is translated to start at (2, 2); windows larger than 31 × 33 are
split (balanced) into blocks that fit, one forward pass per block. Geometry inside a block is unchanged.
267/1,429 windows (111,891 spots) are split. `--he-check` keeps He's own positions, to reproduce He.
Open question for the He arm: He array coords likely also start at 1 → its edge spots hit untrained rows.

**BLEEP he-check** vs the stored He preds: ~90 % spots identical, max diff ≈ 0.22 (= one of 50 retrieved
neighbours swapped), corr > 0.9999. Re-running the *He* driver today gives exactly the same numbers, so the
TCGA path equals the He path; the stored He preds were made on a different GPU/software state. Verified with
`--he-root /workspace/runs/he_bleep_recheck` (100 % identical). BLEEP is not bit-reproducible across hardware.

**TRIPLEX dropout (decision 2026-09-30: dropout OFF at inference).** Upstream `MultiHeadAttention.train()`
(NEXGEM/TRIPLEX `src/model/TRIPLEX/module.py`) calls `super().train(mode)` only when `attn_bias=True`, so
`model.eval()` leaves the global encoder's bias-free attention layers (3 × 6 = 18 modules) and their
`nn.Dropout` in training mode. Every forward was a random draw: re-runs differ by 0.3–0.6 (corr ≈ 0.999),
and no APEG grid reproduces the stored her2st predictions. This is in upstream's own inference too (and its
flash branch also passes `drop_p` unconditionally). `run_triplex_tcga.force_eval` runs `model.eval()` then
clears the flag on the stuck modules → deterministic. Architecture/dropout values match upstream
`config/ST/andersson/TRIPLEX.yaml` exactly. **The her2st, He and Visium TRIPLEX results were made with
dropout on** — re-run them with `force_eval` for a consistent comparison (inference only).

**STFlow he-check** vs stored He preds: max diff 0.7–1.8, corr 0.998–0.9997 (environment drift through the
flow-sampling steps). Against the He driver re-run today: max diff ≤ 3e-5 → the TCGA path equals the He path.
Placeholder check: bit-identical. 1-spot blocks (2 spots in the cohort) are fed as two copies (approximation:
the copies draw different prior noise).

### B.7 After a pod restart (only /workspace persists)

```bash
pip install -q argcomplete pyyaml scikit-image opencv-python-headless pillow scipy h5py openslide-python openslide-bin tqdm pandas matplotlib seaborn scikit-learn statsmodels spams-bin
pip install -q -r /workspace/st_benching/bleep/requirements_bleep.txt
# easydl (Hist2ST) still imports Iterable from collections -- Python >= 3.10 needs collections.abc:
grep -rl "from collections import" /usr/local/lib/python3.12/dist-packages/easydl/ | xargs sed -i -E 's/from collections import (Iterable|Mapping|Sequence|Callable|MutableMapping)\b/from collections.abc import \1/'
```
