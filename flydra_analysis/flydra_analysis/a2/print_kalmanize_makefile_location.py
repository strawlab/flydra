from __future__ import print_function
import pkg_resources
import os


def main():
    RESFILE = pkg_resources.resource_filename(__name__, "Makefile.kalmanize")
    print(RESFILE)
