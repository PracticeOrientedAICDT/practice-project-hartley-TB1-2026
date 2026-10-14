








set -euo pipefail
cd "$(dirname "$0")"
PORT=8000
BUILD_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --build-only) BUILD_ONLY=1 ;;
    [0-9]*) PORT="$arg" ;;
    *) echo "usage: $0 [--build-only] [port]" >&2; exit 1 ;;
  esac
done
python3 scripts/build_site.py --results results --out site
if [ "$BUILD_ONLY" = 1 ]; then
  exit 0
fi
echo "serving site/ at http://localhost:$PORT (Ctrl-C to stop)"
python3 -m http.server "$PORT" --directory site
