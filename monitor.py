#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import sys
import numpy as np
import pyqtgraph as pg
import matplotlib as mpl
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.uic import *
from preprocessing import Preprocessing
from processing import Processing
from multithread import Worker

class Data_Viewer(QMainWindow):
    '''
    A GUI for viewing the latest data file under a designated directory both in time and frequency domains
    '''

    fgcolor = "#23373B"
    bgcolor = "#FAFAFA"
    blue    = "#113285"
    green   = "#1B813E"
    orange  = "#E98B2A"
    red     = "#AB3B3A"
    lut = (mpl.cm.get_cmap("viridis")(np.linspace(0, 1, 256))[:,:3] * 255).astype(np.dtype("u1"))
    pg.setConfigOptions(background=bgcolor, foreground=fgcolor, antialias=True, imageAxisOrder="row-major")
    def font_label(self, string): return "<span style=font-family:RobotoCondensed;font-size:14pt>" + string + "</span>"

    def __init__(self, directory="/home/schospec/Data/"):
        '''
        paint the user interface and establish the signal-socket connections
        directory:      location to be sought for data files
        '''
        super().__init__()
        loadUi("monitor.ui", self) # the .ui file is generated by the Qt creator
        self.thread_pool = QThreadPool()
        self.directory = directory
        self.logarithm = self.rLog.isChecked()
        self.manual = self.rManual.isChecked()
        self.draw_plots()
        self.build_connections()

    def draw_plots(self):
        '''
        draw the time signals separately in in-phase and quadrature parts, also the frequency signals separately in 1d and 2d forms
        '''
        # time plots --- in-phase
        self.plot_i = self.gTimeSeries.addPlot(0, 0)
        self.plot_i.plot([0, 1], [0, 0], pen=self.blue)
        self.plot_i.setLabels(left=self.font_label("In-Phase"), bottom=self.font_label("Time [s]"))
        self.plot_i.setRange(xRange=(0, 1), yRange=(-1, 1))
        # time plots --- quadrature
        self.plot_q = self.gTimeSeries.addPlot(1, 0)
        self.plot_q.plot([0, 1], [0, 0], pen=self.blue)
        self.plot_q.setLabels(left=self.font_label("Quadrature"), bottom=self.font_label("Time [s]"))
        self.plot_q.setRange(xRange=(0, 1), yRange=(-1, 1))
        # dummy data for frequency plots
        self.frequencies = np.array([-1, 0, 1])
        self.times_f = np.array([0, 1])
        self.spectrogram = np.array([[1, 1]])
        self.frame = 0
        self.fill_level = np.floor(np.min(self.spectrogram[self.frame])) if self.logarithm else 0
        # frequency plots --- spectrum
        self.gSpectrum.plot((self.frequencies[:-1]+self.frequencies[1:])/2, self.spectrogram[self.frame], pen=self.orange, fillLevel=self.fill_level, fillBrush=self.orange+"80")
        self.gSpectrum.setLabels(title=self.font_label("Frame # 0"),
                left=self.font_label("Power Spectral Density"), bottom=self.font_label("Frequency − ___ MHz [kHz]"))
        self.gSpectrum.setRange(xRange=(self.frequencies[0], self.frequencies[-1]), yRange=(self.times_f[0], self.times_f[-1]))
        # frequency plots --- spectrogram
        self.img = pg.ImageItem(self.spectrogram)
        self.img.setRect(QRectF(-(self.frequencies[-1]-self.frequencies[0])/2, self.times_f[0], self.frequencies[-1]-self.frequencies[0], self.times_f[-1]-self.times_f[0]))
        self.img.setLookupTable(self.lut)
        self.gSpectrogram.addItem(self.img)
        self.gSpectrogram.setLabels(left=self.font_label("Time [s]"), bottom=self.font_label("Frequency − ___ MHz [kHz]"))
        self.gSpectrogram.setRange(xRange=(self.frequencies[0], self.frequencies[-1]), yRange=(self.times_f[0], self.times_f[-1]))
        # file list
        self.mFileList = QFileSystemModel()
        self.mFileList.setFilter(QDir.Files)
        self.mFileList.setNameFilters(["*.wvd"])
        self.mFileList.setNameFilterDisables(False)
        self.vFileList.setModel(self.mFileList)
        self.vFileList.setRootIndex(self.mFileList.setRootPath(self.directory))

    def build_connections(self):
        '''
        build the signal-slot connections
        '''
        # bind time plots
        def update_range_i():
            self.plot_i.setRange(self.plot_q.getViewBox().viewRect(), padding=0)
        def update_range_q():
            self.plot_q.setRange(self.plot_i.getViewBox().viewRect(), padding=0)
        self.plot_i.sigRangeChanged.connect(update_range_q)
        self.plot_q.sigRangeChanged.connect(update_range_i)
        # bind frequency plots
        self.indicator = self.gSpectrogram.addLine(y=self.times_f[0], bounds=[self.times_f[0], self.times_f[-1]], pen=self.bgcolor, hoverPen=self.red, movable=True)
        def on_dragged(line):
            pos = line.value()
            self.frame = int((pos - self.times_f[0]) / (self.times_f[1] - self.times_f[0]))
            self.frame -= 1 if self.frame == self.times_f.size-1 else 0
            self.fill_level = np.floor(np.min(self.spectrogram[self.frame])) if self.logarithm else 0
            self.gSpectrum.listDataItems()[0].setData((self.frequencies[:-1]+self.frequencies[1:])/2, self.spectrogram[self.frame])
            self.gSpectrum.listDataItems()[0].setFillLevel(self.fill_level)
            current_range = self.gSpectrum.getViewBox().viewRange()[0]
            self.gSpectrum.autoRange()
            self.gSpectrum.setXRange(*current_range, padding=0)
            self.gSpectrum.setTitle(self.font_label("Frame # {:d}".format(self.frame)))
        self.indicator.sigPositionChanged.connect(on_dragged)
        def update_range_spectrum():
            self.gSpectrum.setXRange(*self.gSpectrogram.getViewBox().viewRange()[0], padding=0)
        def update_range_spectrogram():
            self.gSpectrogram.setXRange(*self.gSpectrum.getViewBox().viewRange()[0], padding=0)
        self.gSpectrum.sigRangeChanged.connect(update_range_spectrogram)
        self.gSpectrogram.sigRangeChanged.connect(update_range_spectrum)
        # read coordinates at the cursor
        self.crosshair_h = pg.InfiniteLine(pos=self.fill_level, angle=0, pen=self.fgcolor)
        self.crosshair_v = pg.InfiniteLine(pos=0, angle=90, pen=self.fgcolor)
        self.gSpectrum.addItem(self.crosshair_h, ignoreBounds=True)
        self.gSpectrum.addItem(self.crosshair_v, ignoreBounds=True)
        def on_moved_spectrum(point):
            if self.gSpectrum.sceneBoundingRect().contains(point):
                coords = self.gSpectrum.getViewBox().mapSceneToView(point)
                self.crosshair_h.setValue(coords.y())
                self.crosshair_v.setValue(coords.x())
                self.statusbar.showMessage("δf = {:.5g} kHz, t = {:.5g} s, psd = {:.5g}".format(coords.x(), self.indicator.value(), coords.y()))
        self.gSpectrum.scene().sigMouseMoved.connect(on_moved_spectrum)
        def on_moved_spectrogram(point):
            if self.gSpectrogram.sceneBoundingRect().contains(point):
                coords = self.gSpectrogram.getViewBox().mapSceneToView(point)
                self.crosshair_v.setValue(coords.x())
                self.statusbar.showMessage("δf = {:.5g} kHz, t = {:.5g} s, psd = {:.5g}".format(coords.x(), self.indicator.value(), self.crosshair_h.value()))
        self.gSpectrogram.scene().sigMouseMoved.connect(on_moved_spectrogram)
        def on_clicked(event):
            point = event.scenePos()
            if self.gSpectrogram.sceneBoundingRect().contains(point):
                coords = self.gSpectrogram.getViewBox().mapSceneToView(point)
                self.indicator.setValue(coords.y())
                self.statusbar.showMessage("δf = {:.5g} kHz, t = {:.5g} s, psd = {:.5g}".format(self.crosshair_v.value(), coords.y(), self.crosshair_h.value()))
        self.gSpectrogram.scene().sigMouseClicked.connect(on_clicked)
        # toggle the linear or logarithmic scale, auto or manual refresh mode
        def on_toggled_scale():
            self.logarithm = self.rLog.isChecked()
            if self.logarithm:
                self.spectrogram = np.log10(self.spectrogram)
            else:
                self.spectrogram = np.power(10, self.spectrogram)
            self.img.setImage(self.spectrogram)
            self.fill_level = np.floor(np.min(self.spectrogram[self.frame])) if self.logarithm else 0
            self.gSpectrum.listDataItems()[0].setData((self.frequencies[:-1]+self.frequencies[1:])/2, self.spectrogram[self.frame])
            self.gSpectrum.listDataItems()[0].setFillLevel(self.fill_level)
            current_range = self.gSpectrum.getViewBox().viewRange()[0]
            self.gSpectrum.autoRange()
            self.gSpectrum.setXRange(*current_range, padding=0)
        self.rLin.toggled.connect(on_toggled_scale)
        def on_toggled_refresh():
            self.manual = self.rManual.isChecked()
            if self.manual:
                self.vFileList.activated.connect(selected_file)
                self.mFileList.rowsInserted.disconnect(last_file)
                self.mFileList.rowsRemoved.disconnect(last_file)
            else:
                self.vFileList.clearSelection()
                self.vFileList.activated.disconnect(selected_file)
                self.mFileList.rowsInserted.connect(last_file)
                self.mFileList.rowsRemoved.connect(last_file)
        self.rAuto.toggled.connect(on_toggled_refresh)
        # select a file for analysis
        def selected_file(model_index):
            self.prepare_data(model_index.data())
        def last_file(model_index):
            self.mFileList.sort(3, Qt.DescendingOrder) # sort by the fourth column, i.e. modified time
            QTimer.singleShot(700, lambda: self.prepare_data(model_index.child(0,0).data())) # wait for file transfer completion
        if self.manual:
            self.vFileList.activated.connect(selected_file)
        else:
            self.mFileList.rowsInserted.connect(last_file)
            self.mFileList.rowsRemoved.connect(last_file)
        # Ctrl+W or Ctrl+Q to quit the application
        shortcutW = QShortcut(QKeySequence.Close, self)
        shortcutQ = QShortcut(QKeySequence.Quit, self)
        shortcutW.activated.connect(self.close)
        shortcutQ.activated.connect(self.close)

    def prepare_data(self, data_file):
        '''
        load the data in time domain from disk, and compute the frequency data
        '''
        self.data_file = data_file
        # data in time domain
        preprocessing = Preprocessing(self.directory+self.data_file, verbose=False)
        self.span = preprocessing.span
        self.file_name = preprocessing.fname
        self.center_frequency = preprocessing.center_frequency
        self.times_t, self.iqs = preprocessing.diagnosis(10**5, draw=False) # s, V
        # data in frequency domain, on a spin-off thread
        processing = Processing(self.directory+self.data_file)
        worker = Worker(processing.time_average_2d, window_length=500, n_frame=-1,
                padding_ratio=2, n_offset=0, n_average=10, estimator='p', window="kaiser", beta=14)
        worker.signals.result.connect(self.redraw_plots)
        self.thread_pool.start(worker)

    def redraw_plots(self, args):
        '''
        redraw the in-phase and quadrature plots, as well as the frequency spectrum and spectrogram
        '''
        frequencies, self.times_f, spectrogram = args[:3] # kHz, s, V^2/kHz
        index_l = np.argmin(np.abs(frequencies+self.span/2e3))
        index_r = np.argmin(np.abs(frequencies-self.span/2e3)) + 1
        self.frequencies = frequencies[index_l:index_r] # kHz
        spectrogram = spectrogram[:,index_l:index_r-1] # V^2/kHz
        self.spectrogram = np.log10(spectrogram) if self.logarithm else spectrogram
        self.frame = 0
        self.fill_level = np.floor(np.min(self.spectrogram[self.frame])) if self.logarithm else 0
        self.lFileName.setText(self.file_name)
        # time plots --- in-phase
        self.plot_i.listDataItems()[0].setData(self.times_t, np.real(self.iqs))
        self.plot_i.setRange(xRange=(self.times_t[0], self.times_t[-1]), yRange=(-1, 1))
        # time plots --- quadrature
        self.plot_q.listDataItems()[0].setData(self.times_t, np.imag(self.iqs))
        self.plot_q.setRange(xRange=(self.times_t[0], self.times_t[-1]), yRange=(-1, 1))
        # frequency plots --- spectrum
        self.gSpectrum.listDataItems()[0].setData((self.frequencies[:-1]+self.frequencies[1:])/2, self.spectrogram[self.frame])
        self.gSpectrum.listDataItems()[0].setFillLevel(self.fill_level)
        self.gSpectrum.setLabels(title=self.font_label("Frame # 0"), bottom=self.font_label("Frequency − {:g} MHz [kHz]".format(self.center_frequency/1e6)))
        self.gSpectrum.setRange(xRange=(-self.span/2e3, self.span/2e3), yRange=(self.fill_level, np.max(self.spectrogram[self.frame])))
        # frequency plots --- spectrogram
        self.img.setImage(self.spectrogram)
        self.img.setRect(QRectF(-(self.frequencies[-1]-self.frequencies[0])/2, self.times_f[0], self.frequencies[-1]-self.frequencies[0], self.times_f[-1]-self.times_f[0]))
        self.gSpectrogram.setLabels(bottom=self.font_label("Frequency − {:g} MHz [kHz]".format(self.center_frequency/1e6)))
        self.gSpectrogram.setRange(xRange=(-self.span/2e3, self.span/2e3), yRange=(self.times_f[0], self.times_f[-1]))
        # reset markers
        self.crosshair_h.setValue(self.fill_level)
        self.crosshair_v.setValue(0)
        self.indicator.setValue(self.times_f[0])
        self.indicator.setBounds([self.times_f[0], self.times_f[-1]])
        self.statusbar.clearMessage()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    data_viewer = Data_Viewer()
    data_viewer.show()
    sys.exit(app.exec())
