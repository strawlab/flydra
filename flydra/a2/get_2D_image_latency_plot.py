#!/usr/bin/env python
"""Get Flydra Latency.

Usage:
  flydra_analysis_show_latency FILENAME [options]

Options:
  -h --help     Show this screen.
"""
from docopt import docopt

import tables
import matplotlib.pyplot as plt
import pandas as pd
import sys

import flydra.analysis.result_utils as result_utils

def plot_latency(fname):
    with tables.openFile(fname, mode='r') as h5:
        d = h5.root.data2d_distorted[:]
        camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)

    df = pd.DataFrame(d)
    camn_list = list(df['camn'].unique())
    camn_list.sort()
    fig2 = plt.figure()
    axn=None
    fig3 = plt.figure()
    ax3n = None
    for camn, dfcam in df.groupby('camn'):
        cam_id = camn2cam_id[camn]
        df0 = dfcam[ dfcam['frame_pt_idx']==0 ]
        ts0s = df0['timestamp'].values
        tss = df0['cam_received_timestamp'].values
        frames = df0['frame'].values
        dts = tss-ts0s

        axn = fig2.add_subplot( len(camn_list), 1, camn_list.index(camn)+1,sharex=axn)
        axn.plot(frames,dts,'r.-',label='camnode latency' )
        axn.plot( frames[:-1], ts0s[1:]-ts0s[:-1], 'g.-', label='inter-frame interval' )
        axn.set_xlabel('frame')
        axn.set_ylabel('time (s)')
        axn.text(0,1,cam_id, va='top', ha='left', transform=axn.transAxes)
        if camn_list.index(camn)==0:
            axn.legend()

        ax3n = fig3.add_subplot( len(camn_list), 1, camn_list.index(camn)+1,sharex=ax3n)
        ax3n.plot(frames,ts0s,'g.-', label='calculated triggerbox timestamp')
        ax3n.set_xlabel('frame')
        ax3n.set_ylabel('time (s)')
        ax3n.text(0,1,cam_id, va='top', ha='left', transform=ax3n.transAxes)
        if camn_list.index(camn)==0:
            ax3n.legend()

    plt.show()

def main():
    args = docopt(__doc__)
    fname = args['FILENAME']

    plot_latency(fname)

if __name__=='__main__':
    main()
