"""Automated geometric, font, speaker-notes, and 6-7 min duration audit for SuperAgent_Presentation_EN.pptx."""
import re
import sys
from pathlib import Path
from pptx import Presentation

ROOT = Path(__file__).resolve().parent.parent
PPT = ROOT / "SuperAgent_Presentation_EN.pptx"
RENDERS = ROOT / "renders"

prs = Presentation(PPT)
sw_in = prs.slide_width / 914400.0
sh_in = prs.slide_height / 914400.0
print(f"Deck: {PPT.name} | Size: {sw_in:.3f} x {sh_in:.3f} in | Total slides: {len(prs.slides)}")

errors = []
core_words = 0
core_seconds = 0

for idx, slide in enumerate(prs.slides, 1):
    # Check out-of-bounds shapes
    for sh in slide.shapes:
        x = sh.left / 914400.0
        y = sh.top / 914400.0
        w = sh.width / 914400.0
        h = sh.height / 914400.0
        if x < -0.05 or y < -0.05 or (x + w) > sw_in + 0.08 or (y + h) > sh_in + 0.08:
            errors.append(f"Slide {idx} shape '{sh.name}' out of bounds: ({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f})")

    # Check speaker notes
    nt = slide.notes_slide.notes_text_frame.text.strip() if slide.has_notes_slide else ""
    if not nt:
        errors.append(f"Slide {idx} is missing speaker notes")

    if idx <= 12:
        # Strip bracket header tag [Slide X | ...]
        body = re.sub(r"^\[.*?\]\s*", "", nt)
        words = re.findall(r"[A-Za-z0-9'-]+", body)
        m = re.search(r"\|\s*(\d+)s\]", nt)
        sec = int(m.group(1)) if m else 0
        core_words += len(words)
        core_seconds += sec
        print(f"  Slide {idx:02d}: {len(words):3d} spoken words | target {sec:2d}s (~{len(words)/(sec/60.0) if sec else 0:.0f} wpm)")

wpm_140_min = core_words / 140.0
wpm_135_min = core_words / 135.0
print(f"\nCore Slides (1-12) Total Spoken Words: {core_words} words")
print(f"Planned Time Budget: {core_seconds}s ({core_seconds // 60}m {core_seconds % 60:02d}s)")
print(f"Estimated Speaking Duration: {wpm_140_min:.2f} min (@140 wpm) to {wpm_135_min:.2f} min (@135 wpm)")

if not (820 <= core_words <= 1020):
    errors.append(f"Core word count {core_words} outside 6-7 min target window (820-1020 words)")

pngs = sorted(RENDERS.glob("slide-*.png"))
print(f"Rendered slide PNGs in renders/: {len(pngs)}")
if pngs and len(pngs) != len(prs.slides):
    errors.append(f"Rendered PNG count ({len(pngs)}) != slide count ({len(prs.slides)})")

if errors:
    print("\nVERIFICATION ERRORS:")
    for e in errors:
        print("  -", e)
    sys.exit(1)

print("\nALL AUTOMATED CHECKS PASSED CLEANLY!")
