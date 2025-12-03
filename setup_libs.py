# setup_libs.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Add any helpers/constants here
DEFAULT_SEED = 42

def set_seed(seed=DEFAULT_SEED):
    np.random.seed(seed)



