#!/usr/bin/env bash
# Unattended Hist2ST generalization for hist2st_lopo_833_cp10k:
#   wait for training -> rescore her2st -> Visium -> He et al. -> TCGA, scoring each cohort.
# Stops at the first failed step or failed check (ROUNDTRIP / HE-CHECK / NaN predictions);
# the last lines of the log say where and why.
#
# usage (on the pod):
#   tmux new -d -s gen "bash /workspace/st_benching/hist2st/run_generalization.sh"
#   tail -n 30 /workspace/runs/generalization.log
set -uo pipefail
cd /workspace/st_benching
R=/workspace/runs
RUN=$R/hist2st_lopo_833_cp10k
LOG=$R/generalization.log

say()  { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
fail() { say "STOPPED: $*"; exit 1; }
# run <step log> <cmd...>: output goes to the step log and the main log; stop on a non-zero exit
run() {
    local slog=$1; shift
    say "START $*"
    "$@" 2>&1 | tee -a "$slog" "$LOG"
    local rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] || fail "exit code $rc from: $*"
    say "OK"
}

say "==== Hist2ST generalization, run $RUN ===="

# 0. wait for the A/B re-run to finish. Match only the python process itself (command line
#    starting with python), not shells or tmux clients whose command line merely contains it.
while pgrep -f "^[^ ]*python[^ ]* -m hist2st\.train" > /dev/null; do sleep 60; done
say "no training process left"
for f in fold00_A fold01_B fold02_C fold03_D fold04_E fold05_F fold06_G fold07_H; do
    [ -f "$RUN/$f/model.pt" ] || fail "$RUN/$f/model.pt missing (did the A/B re-run finish?)"
    grep -q '"norm": "cp10k"' "$RUN/$f/run.json" || fail "$RUN/$f/run.json is not a cp10k run"
done
say "all 8 checkpoints present; disk used: $(du -sh /workspace | cut -f1) of 100G"

# 1. final her2st scores (A and B were re-run)
run "$R/score_cp10k.log" python -m hist2st.score --tag hist2st_lopo_833_cp10k

# 2. Visium (her2st held-out round-trip + Visium sections), ~6 min
run "$R/ext_hist2st.log" python external/run_hist2st.py --out "$R/ext_hist2st"
grep -q "ROUNDTRIP PASS" "$R/ext_hist2st.log" || fail "Visium round-trip did not pass"
run "$R/score_ext.log" python external/score_external.py

# 3. He et al.: her2st held-out for the paired delta, then all He sections
run "$R/he_hist2st_f1.log" python external/run_hist2st.py --sections --out "$R/he_hist2st"
grep -q "ROUNDTRIP PASS" "$R/he_hist2st_f1.log" || fail "He F.1 round-trip did not pass"
mkdir -p "$R/he_hist2st"
run "$R/he_hist2st/predict.log" python external/run_hist2st_he.py --out "$R/he_hist2st"
grep -qi "mean PCC nan" "$R/he_hist2st/predict.log" && fail "NaN predictions on He"
run "$R/score_he.log" python external/score_he.py

# 4. TCGA: reproduce He through the TCGA code path first, then all windows (~75 min)
run "$R/tcga_hist2st_hecheck.log" python external/run_hist2st_tcga.py --he-check BC23287_C1
grep -q "HE-CHECK PASS" "$R/tcga_hist2st_hecheck.log" || fail "TCGA he-check did not pass"
run "$R/tcga_hist2st.log" python external/run_hist2st_tcga.py --out "$R/tcga_hist2st"
run "$R/tcga_score.log" python tcga/pseudobulk.py --runs "$R/tcga_hist2st" --out /workspace/results/tcga
run "$R/tcga_score.log" python tcga/score_tcga.py --pb /workspace/results/tcga/pseudobulk_hist2st.tsv.gz

say "ALL DONE; disk used: $(du -sh /workspace | cut -f1) of 100G"
