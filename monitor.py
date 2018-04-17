#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from processing import Processing
from preprocessing import Preprocessing
from matplotlib.widgets import Cursor, Slider, TextBox
from matplotlib.colors import LogNorm
import figure_size as fs
import glob, os
mpl.rcParams["figure.autolayout"] = False
plt.style.use("paper")

fig = plt.figure(figsize=fs.figure_size(scale=1.3, aspect_ratio=3/4))
gs = gridspec.GridSpec(2, 2)
gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,0], hspace=0)

def update(val):
    idx = int(val) - 1
    spectrum.set_ydata(spectrogram[idx])
    banner.set_xy([ [0,times[idx]], [0,times[idx+1]], [1,times[idx+1]], [1,times[idx]], [0,times[idx]] ])
    tb.set_val(idx+1)
    fig.canvas.draw_idle()

def submit(text):
    idx = int(text) - 1
    spectrum.set_ydata(spectrogram[idx])
    banner.set_xy([ [0,times[idx]], [0,times[idx+1]], [1,times[idx+1]], [1,times[idx]], [0,times[idx]] ])
    sld.set_val(idx+1)
    fig.canvas.draw_idle()

new_comer = sorted(glob.iglob("/home/schospec/Data/*.wvh"), key=os.path.getmtime)[-1]
while True:
    ax1 = fig.add_subplot(gs_sub[0,0])
    ax2 = fig.add_subplot(gs_sub[1,0], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[:,1])
    ax4 = fig.add_subplot(gs[1,0], sharex=ax3)

    preprocessing = Preprocessing(new_comer, verbose=False)
    times, signal = preprocessing.load(50000, 0) # s, V
    ax1.plot(times, np.real(signal), color="C0")
    ax2.plot(times, np.imag(signal), color="C0")
    ax1.set_xlim([times.min(), times.max()])
    ax1.set_ylim(-1.1, 1.1)
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.set_ylabel("In-Phase")
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("Quadrature")

    processing = Processing(new_comer)
    frequencies, times, spectrogram, _ = processing.time_average_2d(window_length=500, n_frame=-1, padding_ratio=1, n_offset=0, n_average=10, estimator='p', window=None, beta=None)
    # frequencies, times, spectrogram, _ = processing.time_average_2d(window_length=500, n_frame=500, padding_ratio=1, n_offset=0, n_average=10, estimator='p', window=None, beta=None)
    idxl = np.argmin(np.abs(frequencies+processing.span/2e3))
    idxr = np.argmin(np.abs(frequencies-processing.span/2e3)) + 1
    frequencies = frequencies[idxl:idxr]
    spectrogram = spectrogram[:,idxl:idxr-1]

    pcm = ax3.pcolormesh(frequencies, times, spectrogram)
    banner = ax3.axhspan(times[0], times[1], color="C1", lw=0, alpha=.5)
    ax3.set_xlim([-processing.span/2e3, processing.span/2e3]) # kHz
    ax3.set_ylim([times[0], times[-1]]) # s
    ax3.set_xlabel("Frequency − {:g} MHz [kHz]".format(processing.center_frequency/1e6))
    ax3.set_ylabel("Time [s]")
    cax = fig.colorbar(pcm, ax=ax3)
    cax.set_label("Power Spectral Density")

    frequencies = (frequencies[1:] + frequencies[:-1]) / 2 # kHz
    spectrum, = ax4.plot(frequencies, spectrogram[0], color="C1")
    ax4.set_xlim([-processing.span/2e3, processing.span/2e3]) # kHz
    ax4.set_ylim(ymax=np.max(spectrogram))
    ax4.set_xlabel("Frequency − {:g} MHz [kHz]".format(processing.center_frequency/1e6))
    ax4.set_ylabel("Power Spectral Density")

    fig.suptitle(processing.fname, size=16)
    gs.tight_layout(fig, rect=(0, .04, 1, .96))
    cs3 = Cursor(ax3, useblit=True, c=".6", lw=.6)
    cs4 = Cursor(ax4, useblit=True, c=".6", lw=.6)
    axs = fig.add_axes([.3, .01, .4, .03])
    axt = fig.add_axes([.705, .01, .04, .03])
    sld = Slider(axs, r"Frame #", 1, times.size-1, valfmt="%d", fc="C2", ec="none", alpha=.5)
    tb = TextBox(axt, '', initial='1')
    sld.on_changed(update)
    tb.on_submit(submit)
    fig.canvas.draw_idle()

    while True:
        plt.pause(5) # s
        candidate = sorted(glob.iglob("/home/schospec/Data/*.wvh"), key=os.path.getmtime)[-1]
        if new_comer != candidate:
            new_comer = candidate
            break
    fig.clf()
