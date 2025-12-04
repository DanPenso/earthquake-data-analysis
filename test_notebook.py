#!/usr/bin/env python
"""
Test notebook execution to identify any remaining errors.
"""
import warnings
warnings.filterwarnings('ignore')
import sys
import os

os.chdir("f:\\My Masters\\CT7201 Python Notebooks and Scripting\\Our Project\\Repo\\earthquake-data-analysis")

try:
    from nbconvert.preprocessors import ExecutePreprocessor
    import nbformat
    
    # Read notebook
    nb = nbformat.read('earthquake_analysis.ipynb', as_version=4)
    
    # Create custom preprocessor to track cell execution
    code_cell_count = [0]
    
    class VerboseExecutePreprocessor(ExecutePreprocessor):
        def execute_cell(self, cell, index, store_history=True):
            if cell.get('cell_type') == 'code':
                code_cell_count[0] += 1
                cell_id = cell.get('id', 'unknown')
                source = cell.get('source', '')[:80].replace('\n', ' ')
                print(f"[Code {code_cell_count[0]}] ID={cell_id} | {source}...", flush=True)
                try:
                    result = super().execute_cell(cell, index, store_history)
                    print(f"       -> OK", flush=True)
                    return result
                except Exception as e:
                    print(f"       -> ERROR: {str(e)[:150]}", flush=True)
                    raise
            return super().execute_cell(cell, index, store_history)
    
    # Set up preprocessor with timeout
    ep = VerboseExecutePreprocessor(timeout=600, kernel_name='python3')
    
    # Execute
    print("Starting notebook execution...", flush=True)
    ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})
    print("Notebook executed successfully!", flush=True)
    
except Exception as e:
    print(f"Error during execution: {type(e).__name__}", flush=True)
    print(f"Details: {str(e)[:800]}", flush=True)
    sys.exit(1)
