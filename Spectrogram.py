# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:18:28 2017

@author: SkyBits1
"""

import pyaudio
import threading
import atexit
import numpy as np
import pyqtgraph as pg
import pyaudio
import wave
from PyQt5 import QtCore, QtGui, QtWidgets
FS = 44100 #Hz
CHUNKSZ = 1024 #samples
import PyQt5
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
import pyaudio
import wave
from PyQt5 import QtCore, QtGui, QtWidgets
FS = 44100 #Hz
CHUNKSZ = 1024 #samples
#from PyQt4 import QtGui, QtCore
#PyQt5QtWidgets.QMainWindow,QtGui.QtWidget
from PyQt5 import QtCore, QtGui, QtWidgets
PyQt5.QtWidgets.QWidget
#QtWidgets.QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
class MicrophoneRecorder(object):
    def __init__(self, rate=44100, chunksize=1024):
        self.rate = rate
        self.chunksize = chunksize
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunksize,
                                  stream_callback=self.new_frame)
        self.lock = threading.Lock()
        self.stop = False
        self.frames = []
        atexit.register(self.close)

    def new_frame(self, data, frame_count, time_info, status):
        data = np.fromstring(data, 'int16')
        with self.lock:
            self.frames.append(data)
            if self.stop:
                return None, pyaudio.paComplete
        return None, pyaudio.paContinue
    
    def get_frames(self):
        with self.lock:
            frames = self.frames
            self.frames = []
            return frames
    
    def start(self):
        self.stream.start_stream()

    def close(self):
        with self.lock:
            self.stop = True
        self.stream.close()
        self.p.terminate()

class SpectrogramWidget(pg.PlotWidget):
    #read_collected = QtCore.pyqtSignal(np.ndarray)
    def __init__(self):
        super(SpectrogramWidget, self).__init__()
        self.plotblack()
        self.connect()
        self.update()
        
    def connect(self):
        mic1 = MicrophoneRecorder()
        mic1.start()
        self.mic1 =mic1
        
    def plotblack(self):
        self.img = pg.ImageItem()
        self.addItem(self.img)

        self.img_array = np.zeros((1000, CHUNKSZ//2+1))
        #print ('self.img_array' ,self.img_array)
        #print len(self.img_array)

        # bipolar colormap
        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array([[0,255,255,255], [255,255,0,255], [0,0,0,255], (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)

        # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([-50,40])

        # setup the correct scaling for y-axis
        freq = np.arange((CHUNKSZ/2)+1)/(float(CHUNKSZ)/FS)
        yscale = 1.0/(self.img_array.shape[1]/freq[-1])
        self.img.scale((1./FS)*CHUNKSZ, yscale)

        self.setLabel('left', 'Frequency', units='Hz')
        #print ('frequency', freq)
        #print len(freq)
        #print ('yscale', yscale)
        #print len(yscale)
        #print ('self.img.scale',self.img.scale)
        #print len(self.img.scale)

        # prepare window for later use
        self.win = np.hanning(CHUNKSZ)
        #print (self.win.shape)
        #print ('self.win', self.win)
        #print len(self.win )
        self.show()
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(100)
        # keep reference to timer        
        self.timer = timer

    def update(self):
        frames1 = self.mic1.get_frames()
        #print ('type(frames1)', type(frames1))
        if len(frames1) > 0:
            current_frame = frames1[-1]
            current_frame1 = np.asarray(current_frame)
            #print (current_frame1.shape)
        # normalized, windowed frequencies in data chunk
            spec = np.fft.rfft(current_frame1*self.win) / CHUNKSZ
        #print ('spec',spec)
        #print len(spec)
        # get magnitude 
            psd = abs(spec)
        #print ('psd1',psd)
        # convert to dB scale
            psd = 20 * np.log10(psd)
        #print ('psd2', psd)
        #print len(psd)

        # roll down one and replace leading edge with new data
            self.img_array = np.roll(self.img_array, -1, 0)
            self.img_array[-1:] = psd
        #print ('self.img_array', self.img_array)
        #print len(self.img_array)
            self.img.setImage(self.img_array, autoLevels=False)
class MplFigure(object):
    def __init__(self, parent):
        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, parent)
class LiveFFTWidget(QtWidgets.QWidget):
    def __init__(self):
        super(LiveFFTWidget, self).__init__()
        #QtWidgets.QWidget.__init__(self)
        
        # customize the UI
        self.initUI()
        
        # init class data
        self.initData()       
        # connect slots
        self.connectSlots()
        
        # init MPL widget
        self.initMplWidget()
        
    def initUI(self):

        #hbox_gain = QtWidgets.QHBoxLayout()
        #autoGain = QtWidgets.QLabel('Auto gain for frequency spectrum')
        #autoGainCheckBox = QtWidgets.QCheckBox(checked=True)
        #hbox_gain.addWidget(autoGain)
        #hbox_gain.addWidget(autoGainCheckBox)
        
        # reference to checkbox
        #self.autoGainCheckBox = autoGainCheckBox
        
        #hbox_fixedGain = QtWidgets.QHBoxLayout()
        #fixedGain = QtWidgets.QLabel('Manual gain level for frequency spectrum')
        #fixedGainSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        #hbox_fixedGain.addWidget(fixedGain)
        #hbox_fixedGain.addWidget(fixedGainSlider)

        #self.fixedGainSlider = fixedGainSlider

        vbox = QtWidgets.QVBoxLayout()

        #vbox.addLayout(hbox_gain)
        #vbox.addLayout(hbox_fixedGain)

        # mpl figure
        self.main_figure = MplFigure(self)
        vbox.addWidget(self.main_figure.toolbar)
        vbox.addWidget(self.main_figure.canvas)
        
        self.setLayout(vbox) 
        
        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('LiveFFT')    
        self.show()
        # timer for callbacks, taken from:
        # http://ralsina.me/weblog/posts/BB974.html
        timer = QtCore.QTimer()
        timer.timeout.connect(self.handleNewData)
        timer.start(20)
        # keep reference to timer        
        self.timer = timer
        
     
    def initData(self):
        mic = MicrophoneRecorder()
        mic.start()  

        # keeps reference to mic        
        self.mic = mic
        
        # setup the correct scaling for y-axis
        
        # computes the parameters that will be used during plotting
        #self.freq_vect = np.fft.rfftfreq(mic.chunksize, 
                                        #1./mic.rate)
         
        self.time_vect = np.arange(mic.chunksize, dtype=np.float32) / mic.rate * 1000


    def connectSlots(self):
        pass
    
    def initMplWidget(self):
        """creates initial matplotlib plots in the main window and keeps 
        references for further use"""
        # top plot
        self.ax_top = self.main_figure.figure.add_subplot(211)
        self.ax_top.set_ylim(-32768, 32768)
        self.ax_top.set_xlim(0, self.time_vect.max())
        self.ax_top.set_xlabel(u'time (ms)', fontsize=6)

        # bottom plot
        #self.ax_bottom = self.main_figure.figure.add_subplot(212)
        #self.ax_bottom.set_ylim(0, 1)
        #self.ax_bottom.set_xlim(0, self.freq_vect.max())
        #self.ax_bottom.set_xlabel(u'frequency (Hz)', fontsize=6)
        # line objects        
        self.line_top, = self.ax_top.plot(self.time_vect, 
                                         np.ones_like(self.time_vect))
        
        #self.line_bottom, = self.ax_bottom.plot(self.freq_vect,
                                               #np.ones_like(self.freq_vect))
                                               
        #self.line_bottom, = self.ax_bottom.plot(self.freq_vect,
                                              #.ones_like(self.freq_vect1))
                                               
    def handleNewData(self):
        """ handles the asynchroneously collected sound chunks """        
        # gets the latest frames        
        frames = self.mic.get_frames()
        self.win = np.hanning(1024)
        
        if len(frames) > 0:
            # keeps only the last frame
            current_frame = frames[-1]
            # plots the time signal
            self.line_top.set_data(self.time_vect, current_frame)
            # computes and plots the fft signal            
            #fft_frame = np.fft.rfft(current_frame)
            #if self.autoGainCheckBox.checkState() == QtCore.Qt.Checked:
            #fft_frame1 /= np.abs(fft_frame1).max()
            #else:
                #fft_frame *= (1 + self.fixedGainSlider.value()) / 5000000.
                #print(np.abs(fft_frame).max())
            #self.line_bottom.set_data(self.freq_vect, np.abs(fft_frame1))            
            
            # refreshes the plots
            self.main_figure.canvas.draw()
import sys 
from PyQt5.QtWidgets import QApplication
if __name__ == "__main__": 
    app = QApplication(sys.argv) 
    Window = LiveFFTWidget()
    w= SpectrogramWidget()
    sys.exit(app.exec_())