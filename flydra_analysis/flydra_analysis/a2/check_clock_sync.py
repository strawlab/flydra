import tables
import numpy
import pylab

if __name__ == "__main__":
    filename = "DATA20070214_192124.h5"
    kresults = tables.open_file(filename, mode="r")

    hci = kresults.root.host_clock_info
    tbl = hci[:]
    kresults.close()

    hostnames = tbl["remote_hostname"]
    uhostnames = numpy.unique(hostnames)
    uhostnames.sort()

    for hostname in uhostnames:
        cond = hostnames == hostname
        start = tbl["start_timestamp"][cond]
        stop = tbl["stop_timestamp"][cond]
        remote = tbl["remote_timestamp"][cond]
        max_measurement_error = stop - start
        max_clock_diff = remote - start

        pylab.figure()
        ax = pylab.subplot(3, 1, 1)
        pylab.plot((start - start[0]) / 60.0, max_clock_diff * 1e3)
        pylab.setp(ax, "ylim", (-10, 10))
        pylab.title(hostname)
        pylab.xlabel("time (min)")
        pylab.ylabel("max clock diff (ms)")

        ax = pylab.subplot(3, 1, 2)
        pylab.plot(max_measurement_error * 1e3, max_clock_diff * 1e3, ".")
        pylab.setp(pylab.gca(), "ylim", (-10, 10))
        pylab.xlabel("max measurement error (ms)")
        pylab.ylabel("max clock diff (ms)")

        ax = pylab.subplot(3, 1, 3)
        bins = numpy.linspace(-11, 11, 50)
        pylab.hist(max_clock_diff * 1e3, bins=bins)
        pylab.setp(ax, "ylim", (0, 3500))
        pylab.xlabel("max clock diff (ms)")
        pylab.ylabel("N occurances")

        sortedidxs = max_clock_diff.argsort()
        earliest = max_clock_diff[sortedidxs[0]]
        latest = max_clock_diff[sortedidxs[-1]]

        if earliest < -latest:
            worst = earliest
            idx = sortedidxs[0]
        else:
            worst = latest
            idx = sortedidxs[-1]

        meas_err_at_worst = max_measurement_error[idx]

        pylab.text(
            0.5,
            0.95,
            "mean = %.3f +/- %.1f ms (N=%d), worst=%.3f ms (err %.3f ms)"
            % (
                numpy.mean(max_clock_diff) * 1e3,
                numpy.std(max_clock_diff) * 1e3,
                len(max_clock_diff),
                worst * 1e3,
                meas_err_at_worst * 1e3,
            ),
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax.transAxes,
        )

    pylab.show()
