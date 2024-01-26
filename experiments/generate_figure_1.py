import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Use the most recent experiment
RESULTS_DIR = "results"
INPUT_DIR = f"{RESULTS_DIR}/{sorted(os.listdir(RESULTS_DIR))[-1]}"
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


LOSS_DATA = {}
L2_DATA = {}
for use_quasi_mc in [True, False]:
    for use_modified_loss in [True, False]:
        filename = f"qmc={use_quasi_mc}_modloss={use_modified_loss}"
        output_name = f"USE_QMC={use_quasi_mc}, USE_MOD_LOSS={use_modified_loss}"
        data = np.load(f"{INPUT_DIR}/{filename}.npz")
        LOSS_DATA[output_name] = data["loss_function_1"]
        L2_DATA[output_name] = data["l2_residual_1"]


# L2 Residual Plot on a log y scale
for name, ts in L2_DATA.items():
    plt.semilogy(ts, label=name)
plt.legend(loc=1)
plt.title("L2 Residual")
plt.savefig(f"{OUTPUT_DIR}/l2_residual.png", bbox_inches="tight")
plt.clf()


# Plot a smoothened loss
WINDOW_SIZE=30
for name, ts in LOSS_DATA.items():
    plt.plot(pd.Series(ts).rolling(window=WINDOW_SIZE).mean(), label=name)
plt.legend(loc=1)
plt.title("Smoothened Loss")
plt.savefig(f"{OUTPUT_DIR}/smoothened_loss.png", bbox_inches="tight")
plt.clf()