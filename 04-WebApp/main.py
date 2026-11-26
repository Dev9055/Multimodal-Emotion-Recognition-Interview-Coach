#!/usr/bin/env python3

import importlib.util
import os
import subprocess
import sys

def _ensure_tensorflow_python():
    if importlib.util.find_spec("tensorflow") is not None:
        return
    if os.environ.get("_MER_PYTHON_CHECKED"):
        return

    current = os.path.realpath(sys.executable)
    candidates = [
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3",
    ]
    for py in candidates:
        if not os.path.isfile(py) or os.path.realpath(py) == current:
            continue
        check = subprocess.run(
            [py, "-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('tensorflow') else 1)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if check.returncode == 0:
            print(f"TensorFlow not found in {sys.executable}; switching to {py}...")
            os.environ["_MER_PYTHON_CHECKED"] = "1"
            os.execv(py, [py, os.path.abspath(__file__), *sys.argv[1:]])

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    _ensure_tensorflow_python()
    from app import main
    main()
