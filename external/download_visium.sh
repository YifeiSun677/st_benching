#!/usr/bin/env bash
# Stage 1 -- HEAD-check every URL, then download. Nothing downloads unless all say 200.
# usage: bash external/download_visium.sh          (re-runnable; skips files already present)
set -uo pipefail
mkdir -p /workspace/ext/raw && cd /workspace/ext/raw
B=https://cf.10xgenomics.com/samples/spatial-exp
cat > urls.tsv <<'TSV'
I1	1.1.0/V1_Breast_Cancer_Block_A_Section_1/V1_Breast_Cancer_Block_A_Section_1	_image.tif
I2	1.1.0/V1_Breast_Cancer_Block_A_Section_2/V1_Breast_Cancer_Block_A_Section_2	_image.tif
J1	2.0.0/CytAssist_FFPE_Human_Breast_Cancer/CytAssist_FFPE_Human_Breast_Cancer	_tissue_image.tif
K1	1.3.0/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer	_image.tif
TSV
fail=0
while IFS=$'\t' read -r sec stem img; do
  for suf in _filtered_feature_bc_matrix.h5 _spatial.tar.gz "$img"; do
    code=$(curl -sIL -o /dev/null -w '%{http_code}' "$B/$stem$suf")
    echo "$code  $sec  $stem$suf"
    [ "$code" = "200" ] || fail=1
  done
done < urls.tsv
if [ $fail = 1 ]; then
  echo "STOP: fix the non-200 lines in urls.tsv (copy the link from the dataset page), then rerun."; exit 1
fi
while IFS=$'\t' read -r sec stem img; do
  mkdir -p "$sec"
  for suf in _filtered_feature_bc_matrix.h5 _spatial.tar.gz "$img"; do
    f="$sec/$(basename "$stem$suf")"
    [ -s "$f" ] || curl -fL --retry 3 -o "$f" "$B/$stem$suf"
  done
  tar --no-same-owner -xzf "$sec"/*_spatial.tar.gz -C "$sec"
done < urls.tsv
sha256sum */*.h5 */*.tif */*.tar.gz > MANIFEST.sha256
du -sh */
