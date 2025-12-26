#!/usr/bin/env python3
"""Launcher for the Streamlit app (uses .venv if present).

Run without arguments to start the app, or pass extra Streamlit args
e.g. `python run.py --server.port 8502`.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(HERE, "app", "streamlit_app.py")

def find_python_in_venv():
    venv = os.path.join(HERE, ".venv")
    if os.path.isdir(venv):
        if os.name == "nt":
            candidate = os.path.join(venv, "Scripts", "python.exe")
        else:
            candidate = os.path.join(venv, "bin", "python")
        if os.path.exists(candidate):
            return candidate
    return sys.executable

def main():
    if not os.path.exists(APP_PATH):
        print(f"Streamlit app not found: {APP_PATH}")
        return 2

    py = find_python_in_venv()
    cmd = [py, "-m", "streamlit", "run", APP_PATH]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    print("Starting Streamlit with:", " ".join(cmd))
    return subprocess.call(cmd)

if __name__ == "__main__":
    raise SystemExit(main())
