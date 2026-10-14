#!/usr/bin/env bash
# Download and unpack the KITTI raw drives listed in configs/default.yaml.
#
# KITTI raw data requires accepting the KITTI license: register at
# https://www.cvlibs.net/datasets/kitti/raw_data.php before downloading.
# The URL pattern below is the one used by KITTI's official raw_data_downloader.sh.
# If it changes, download the "synced+rectified data" zip of each drive from the
# KITTI page by hand and unzip it into data/kitti_raw/.
#
# Disk space: long drives are large (several GB each, mostly camera images).
# After unpacking you may delete image_0*/ folders; FreshScan only needs
# velodyne_points/ and oxts/.
#
# Usage (from the repository root):  bash scripts/00_download_kitti.sh [--keep-images]

set -e
cd "$(dirname "$0")/.."
OUT=data/kitti_raw
mkdir -p "$OUT"
KEEP_IMAGES=0
[ "$1" == "--keep-images" ] && KEEP_IMAGES=1

DRIVES=$(python3 - <<'EOF'
import yaml
c = yaml.safe_load(open("configs/default.yaml"))
print(" ".join(c["calibration_drives"] + c["test_drives"]))
EOF
)

for d in $DRIVES; do
  zip="${d}_sync.zip"
  url="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${d}/${zip}"
  if [ -d "$OUT/${d:0:10}/${d}_sync" ]; then
    echo "[skip] $d already present"; continue
  fi
  echo "[get ] $d"
  wget -c -q --show-progress -O "$OUT/$zip" "$url"
  unzip -q -o "$OUT/$zip" -d "$OUT"
  rm -f "$OUT/$zip"
  if [ "$KEEP_IMAGES" -eq 0 ]; then
    rm -rf "$OUT/${d:0:10}/${d}_sync"/image_0*
  fi
done
echo "Done. Drives are in $OUT"
