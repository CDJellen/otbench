from typing import Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_predictions(y_true: Union[pd.Series, pd.DataFrame], y_pred: Union[pd.Series, pd.DataFrame, np.ndarray],
                     *args) -> None:
    """Plot the actual and predicted values."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.autofmt_xdate(rotation=45)

    # change the background color
    plt.rcParams["axes.facecolor"] = "#faf9f6"
    plt.rcParams["font.family"] = "sans serif"

    # add grid lines
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    # add axis labels
    ax.set_xlabel(
        "Time",
        fontsize=15,
    )
    if y_pred[0] < 0:
        ax.set_ylabel(r"$\log_{10} C_n^2$", fontsize=12)
    else:
        ax.set_ylabel(r"$C_n^2$", fontsize=12)

    # plot the actual and predicted values
    ax.scatter(y=y_true.values, x=y_true.index, label="actual", alpha=0.9, c="#00356B", marker="o")
    ax.scatter(y=y_pred, x=y_true.index, label="predicted", alpha=0.9, c="#C90016", marker=".")
    for i in range(len(args)):
        if isinstance(args[i], pd.Series) or (isinstance(args[i], np.ndarray) and len(args[i]) == len(y_true)):
            ax.scatter(y=args[i], x=y_true.index, label="model {}".format(i + 1), alpha=0.9, marker=".")

    # add legend
    ax.legend(
        loc="best",
        fontsize=10,
        frameon=True,
        framealpha=0.5,
    )

    # return the plot
    return fig, ax
