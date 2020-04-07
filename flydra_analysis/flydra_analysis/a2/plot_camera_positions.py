import flydra_core.reconstruct
import pymvg
import pymvg.multi_camera_system


from pymvg.plot_utils import plot_system
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys, os


def show_flydra_cal(flydra_filename):
    R = flydra_core.reconstruct.Reconstructor(flydra_filename)
    system = R.convert_to_pymvg(ignore_water=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plot_system(ax, system)
    plt.show()


def main():
    flydra_filename = sys.argv[1]
    show_flydra_cal(flydra_filename)


if __name__ == "__main__":
    main()
