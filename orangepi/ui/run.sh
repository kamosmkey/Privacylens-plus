#!/usr/bin/env bash
set -euo pipefail

ui_dir="$(cd "$(dirname "$0")" && pwd)"
default_python="$ui_dir/../rknn-env/bin/python"
python_bin="${UI_PYTHON:-$default_python}"

if [[ ! -x "$python_bin" ]]; then
    echo "Python environment not found: $python_bin" >&2
    echo "Set UI_PYTHON to the Python executable that contains the UI dependencies." >&2
    exit 1
fi

cd "$ui_dir"
exec "$python_bin" app.py "$@"
