#!/usr/bin/env python3
"""
Launcher for the pytink model viewer Streamlit app.

Run directly:
    streamlit run src/pytink/viewer_app.py

Or via the installed console script, which re-execs this file under
`streamlit run`:
    pytink-viewer
"""
import sys
from pathlib import Path


def main():
    """Console-script entry point: re-exec this file under `streamlit run`."""
    from streamlit.web import cli as stcli

    script_path = str(Path(__file__).resolve())
    sys.argv = ["streamlit", "run", script_path, "--"] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == "__main__":
    # Reached when Streamlit executes this file as the target script.
    from pytink.analysis import ModelViewer

    ModelViewer().render()
