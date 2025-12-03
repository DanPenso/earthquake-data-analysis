# Earthquake Data Analysis — Combined Script

This repository contains a combined helper script `earthquake.py` that merges the original `setup_libs.py`, `debug_setup.py`, and `test_setup.py` for simpler imports and a single point of maintenance.

Usage

- In notebooks or scripts, import the helper module as:

```python
import earthquake as libs

# then use
libs.availability()
libs.apply_default_plot_style()
```

- Run the built-in diagnostic/test output from the command line (uses current Python environment):

```powershell
& "C:\Users\Dinis PC\anaconda3\envs\earthquake\python.exe" "path\to\earthquake.py"
```

Notes

- `HAS_SKLEARN` may be False on Python 3.14 because some `scikit-learn` builds are not yet compatible.
- The debug/test code runs only when `earthquake.py` is executed directly.

Files changed/created

- `earthquake.py` — combined module (created)
- `test_setup.py` — updated to import `earthquake` (modified)
- `README.md` — this file (created)

If you want, I can:
- Commit these changes to git with a concise commit message.
- Update the notebook to import `earthquake` instead of `setup_libs`.
- Remove or archive the original `setup_libs.py` file to avoid duplicate names.
