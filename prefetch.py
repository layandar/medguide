"""Cache the official label for your demo drugs so the demo works offline.

Usage:  python prefetch.py ibuprofen paracetamol amoxicillin metformin
(openFDA uses US names: acetaminophen instead of paracetamol, albuterol instead of salbutamol.)
"""

import sys

from retrieval import fetch_label, build_context

names = sys.argv[1:] or ["ibuprofen", "acetaminophen", "amoxicillin", "metformin"]
for n in names:
    label = fetch_label(n)
    if not label:
        print(f"[miss] {n}: no label found (or no network)")
        continue
    ctx, sections = build_context(label)
    print(f"[ok]   {n}: {len(ctx)} chars from {', '.join(sections)}")
