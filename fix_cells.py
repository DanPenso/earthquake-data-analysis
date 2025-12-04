#!/usr/bin/env python
"""Remove the misplaced visualization cells"""
import json

with open('earthquake_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])

# Find and remove cells containing viz_clean calls that are too early
cells_to_remove = []

for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        # These are the problematic cells in the early section
        if i < 30 and 'viz_clean.' in source:  # viz_clean shouldn't appear before cell 30
            print(f"Marking cell {i} for removal: {source[:60]}...")
            cells_to_remove.append(i)

# Remove cells in reverse order to maintain indices
for i in reversed(cells_to_remove):
    print(f"Removing cell {i}")
    del cells[i]

# Write back
with open('earthquake_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"\nRemoved {len(cells_to_remove)} cells")
print(f"Total cells now: {len(cells)}")
