from __future__ import print_function
import sys
import tables as PT
import tables
import flydra_core.data_descriptions

NewInfo2D = flydra_core.data_descriptions.Info2D


def main():
    filename = sys.argv[1]

    results = tables.open_file(filename, mode="r+")
    chunksize = 10000
    start_idx = 0
    results.root.data2d_distorted._f_rename("orig_data2d")
    orig_data2d = results.root.orig_data2d
    data2d_distorted = results.create_table(
        results.root,
        "data2d_distorted",
        NewInfo2D,
        "2d data",
        expectedrows=orig_data2d.nrows,
    )

    # fast way would start something like this:
    ##    while start_idx < orig_data2d.nrows:
    ##        all_data2d = orig_data2d.read(start=start_idx,
    ##                                      stop=start_idx+chunksize,
    ##                                      flavor='numpy')
    ##        start_idx += chunksize

    this_frame = -1
    this_camn = -1
    this_idx = 0
    new_row = data2d_distorted.row
    for rownum, row in enumerate(orig_data2d):
        if rownum % 1000 == 0:
            print("row %d of %d" % (rownum, orig_data2d.nrows))
        if row["frame"] != this_frame:
            this_idx = 0
            this_frame = row["frame"]
        if row["camn"] != this_camn:
            this_idx = 0
            this_camn = row["camn"]

        new_row["camn"] = this_camn
        new_row["frame"] = this_frame
        for attr in (
            "timestamp",
            "x",
            "y",
            "area",
            "slope",
            "eccentricity",
            "p1",
            "p2",
            "p3",
            "p4",
        ):
            new_row[attr] = row[attr]
        new_row["frame_pt_idx"] = this_idx
        new_row.append()
        this_idx += 1
    orig_data2d._f_remove()
    print("you may want to call ptrepack!")


if __name__ == "__main__":
    main()
