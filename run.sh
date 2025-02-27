#!/bin/sh
set -e
cd -- "$(dirname -- "$BASH_SOURCE")"
python3 -m venv .venv
source .venv/bin/activate
python3 video_transcriber.py
