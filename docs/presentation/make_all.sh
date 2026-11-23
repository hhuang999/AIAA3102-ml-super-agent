#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"

echo "[1/5] Rendering custom W3C vector SVG diagrams..."
"$PY" scripts/make_svg_diagrams.py

echo "[2/5] Generating 300-DPI Matplotlib benchmark plots..."
"$PY" scripts/make_plots.py

echo "[3/5] Building SuperAgent_Presentation_EN.pptx on HKUST(GZ) template..."
"$PY" build_ppt.py

echo "[4/5] Exporting PDF via Microsoft PowerPoint & rendering slide PNGs..."
bash scripts/export_pdf.sh SuperAgent_Presentation_EN.pptx SuperAgent_Presentation_EN.pdf
rm -f renders/slide-*.png
/opt/homebrew/bin/pdftoppm -r 140 -png SuperAgent_Presentation_EN.pdf renders/slide

echo "[5/5] Running automated bounds & 6-7 min script verification..."
"$PY" scripts/verify_deck.py
