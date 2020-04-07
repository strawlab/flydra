from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_skipped_frames(fname):
    orig_store = pd.HDFStore(fname, mode="r")
    df = orig_store["skipped_info"]
    values = df["duration"].values
    worst_row_idx = np.argmax(values)
    print("worst row:")
    print(df.iloc[worst_row_idx])
    orig_store.close()
    plt.hist(values, 100)
    plt.xlabel("N frames skipped")
    plt.ylabel("count")
    plt.show()


def main():
    fname = sys.argv[1]
    plot_skipped_frames(fname)


if __name__ == "__main__":
    main()
