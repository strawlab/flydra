from __future__ import print_function
import sys

import numpy as np
import pandas as pd


def main():
    fn = sys.argv[1]

    store = pd.HDFStore(fn, "r")

    camids = {}
    for i, s in store["cameras"].iterrows():
        camids[s["cam_id"]] = s["camn"]

    pts = {}
    for camid in camids:
        pts[camid] = store.select("/reprojection", where="camn = %d" % camids[camid])[
            "dist"
        ]

    for camid in sorted(camids):
        dist = pts[camid]
        _mean = dist.mean()
        _std = dist.std()
        _max = dist.max()
        _med = dist.quantile(0.5)
        _lq = dist.quantile(0.1)
        _uq = dist.quantile(0.9)
        print(
            camid,
            "mean",
            _mean,
            "std",
            _std,
            "max",
            _max,
            "med",
            _med,
            "10%",
            _lq,
            "90%",
            _uq,
        )


if __name__ == "__main__":
    main()
