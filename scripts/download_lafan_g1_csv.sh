#!/usr/bin/env bash
set -euo pipefail

# Download all CSV files under `g1/` from the LAFAN1 Retargeting dataset
# without requiring Hugging Face CLI.

REPO="lvhaidong/LAFAN1_Retargeting_Dataset"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="datasets/csv/lafan"
BASE_URL="${HF_BASE_URL:-https://huggingface.co}"
MAX_RETRIES=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  scripts/download_lafan_g1_csv.sh [OUT_DIR] [--base-url URL] [--retries N]

Examples:
  scripts/download_lafan_g1_csv.sh
  scripts/download_lafan_g1_csv.sh datasets/csv/lafan --base-url https://hf-mirror.com
  HF_BASE_URL=https://hf-mirror.com scripts/download_lafan_g1_csv.sh

Notes:
  - Relative OUT_DIR is resolved against repository root.
  - Existing files are kept and will be skipped.
EOF
      exit 0
      ;;
    --*)
      echo "[ERROR] Unknown option: $1" >&2
      exit 1
      ;;
    *)
      OUT_DIR="$1"
      shift
      ;;
  esac
done

if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$REPO_ROOT/$OUT_DIR"
fi

BASE_URL="${BASE_URL%/}"

mkdir -p "$OUT_DIR"

echo "[INFO] Output directory: $OUT_DIR"
echo "[INFO] Dataset: $REPO"
echo "[INFO] Base URL: $BASE_URL"
echo "[INFO] Retries per file: $MAX_RETRIES"

if command -v uv >/dev/null 2>&1; then
  PY_CMD=(uv run python)
elif command -v python3 >/dev/null 2>&1; then
  PY_CMD=(python3)
else
  echo "[ERROR] python3 not found. Please install Python 3 first." >&2
  exit 1
fi

"${PY_CMD[@]}" - "$OUT_DIR" "$REPO" "$BASE_URL" "$MAX_RETRIES" <<'PY'
import json
import os
import ssl
import sys
import time
import urllib.request
from urllib.error import URLError, HTTPError

out_dir = sys.argv[1]
repo = sys.argv[2]
base_url = sys.argv[3].rstrip("/")
max_retries = int(sys.argv[4])

if max_retries < 1:
    raise SystemExit("--retries must be >= 1")

api = f"{base_url}/api/datasets/{repo}/tree/main/g1?recursive=1"
base = f"{base_url}/datasets/{repo}/resolve/main"

ssl_context = ssl.create_default_context()
opener = urllib.request.build_opener()
opener.addheaders = [("User-Agent", "mimic-clai-downloader/1.0")]

with opener.open(api, timeout=60) as response:
    items = json.load(response)

csv_paths = [item.get("path", "") for item in items if item.get("path", "").endswith(".csv")]
if not csv_paths:
    raise SystemExit("No CSV files found under g1/. Please check dataset path or network.")

print(f"Found {len(csv_paths)} CSV files under g1/.")

for idx, rel_path in enumerate(sorted(csv_paths), start=1):
    filename = os.path.basename(rel_path)
    url = f"{base}/{rel_path}?download=true"
    dst = os.path.join(out_dir, filename)

    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[{idx}/{len(csv_paths)}] Skipped existing: {filename}")
        continue

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url)
            with opener.open(req, timeout=120) as resp, open(dst, "wb") as f:
                f.write(resp.read())
            print(f"[{idx}/{len(csv_paths)}] Downloaded: {filename}")
            last_err = None
            break
        except (URLError, HTTPError, ssl.SSLError, TimeoutError, ConnectionError) as err:
            last_err = err
            if os.path.exists(dst):
                try:
                    os.remove(dst)
                except OSError:
                    pass
            if attempt < max_retries:
                delay = min(2 ** attempt, 12)
                print(
                    f"[{idx}/{len(csv_paths)}] Retry {attempt}/{max_retries - 1} for {filename} "
                    f"after error: {err}. Waiting {delay}s..."
                )
                time.sleep(delay)

    if last_err is not None:
        raise SystemExit(f"Failed to download {filename} after {max_retries} attempts: {last_err}")

print(f"Done. Files saved to: {out_dir}")
PY
