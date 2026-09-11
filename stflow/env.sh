# stflow_port/env.sh -- source this in every new shell on the pod:
#     source /workspace/st_benching/stflow_port/env.sh
# Edit HER2ST_ROOT if `find /workspace -maxdepth 4 -type d -name ST-cnts` shows a different place.
export WORKSPACE=${WORKSPACE:-/workspace}
export ST_BENCH=${ST_BENCH:-$WORKSPACE/st_benching}
export STFLOW_REPO=${STFLOW_REPO:-$WORKSPACE/repos/STFlow}
export HER2ST_ROOT=${HER2ST_ROOT:-$WORKSPACE/her2st/data}
export PANEL=${PANEL:-$ST_BENCH/panels/panel_833.txt}
export UNI_CKPT=${UNI_CKPT:-$WORKSPACE/weights/uni/pytorch_model.bin}
export STFLOW_CACHE=${STFLOW_CACHE:-$WORKSPACE/stflow_cache}
export RUNS_ROOT=${RUNS_ROOT:-$WORKSPACE/runs}
export HF_HOME=${HF_HOME:-$WORKSPACE/hf_cache}              # HF downloads survive pod deletion
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$WORKSPACE/pip_cache}       # makes the next pod rebuild faster
export CACHE_TAG=${CACHE_TAG:-uni_v1_hest112}   # which feature cache train.py reads
export PYTHONUNBUFFERED=1
