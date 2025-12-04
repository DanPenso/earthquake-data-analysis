#!/usr/bin/env python
"""Replace engineer_features() calls with dataset.engineer_features()"""
import json

with open('earthquake_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
fixed_count = 0

# Find and fix cells calling engineer_features() instead of dataset.engineer_features()
for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'code':
        source_list = cell.get('source', [])
        if isinstance(source_list, str):
            source_list = [source_list]
        
        source = ''.join(source_list)
        
        # Check if it has standalone engineer_features() calls
        if 'engineer_features(' in source and 'dataset.engineer_features' not in source:
            # Fix it
            new_source = source.replace('engineer_features(cleaned_eq_df)', 'dataset.engineer_features()')
            
            # Check if anything changed
            if new_source != source:
                print(f"Fixed cell {i}: {cell.get('id')}")
                print(f"  Before: {source[:60]}...")
                print(f"  After: {new_source[:60]}...")
                
                # Update the source
                if isinstance(cell.get('source'), str):
                    cell['source'] = new_source
                else:
                    cell['source'] = new_source.split('\n')
                
                fixed_count += 1

# Write back
if fixed_count > 0:
    with open('earthquake_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"\nFixed {fixed_count} cells")
else:
    print("No cells to fix")
