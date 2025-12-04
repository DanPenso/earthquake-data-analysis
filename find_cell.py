#!/usr/bin/env python
"""Find cells with engineer_features() standalone function calls"""
import json

with open('earthquake_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
code_cell_num = 0

# Find code cells with engineer_features calls
for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'code':
        code_cell_num += 1
        source = ''.join(cell.get('source', []))
        
        # Look for standalone engineer_features function call (not dataset.engineer_features)
        if 'engineer_features(' in source and 'dataset.engineer_features' not in source:
            print(f"\nCode Cell {code_cell_num} (Notebook cell {i})")
            print(f"ID: {cell.get('id')}")
            print(f"Content:\n{source[:200]}...")

