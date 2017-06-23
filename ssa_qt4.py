import sys
#this is for qt5, which didn't want to compile
#from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QLineEdit, QLabel, QFileDialog, QHBoxLayout, QFrame, QGridLayout
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

#for qt4 try this
from PyQt4.QtGui import QDialog, QApplication, QPushButton, QVBoxLayout, QLineEdit, QLabel, QFileDialog, QHBoxLayout, QFrame, QGridLayout
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas


import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import hankel
from sklearn.utils.extmath import randomized_svd
from scipy.signal import detrend


#this info based on the stackoverflow post here
#http://stackoverflow.com/questions/12459811/how-to-embed-matplotib-in-pyqt-for-dummies

#import random

class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure()
        #self.figure.tight_layout()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        #self.toolbar = NavigationToolbar(self.canvas, self)
        
        
        #make a file open dialog
        #self.dialog = QDialog(d)
        self.btn = QPushButton("Open file")
        self.btn.clicked.connect(self.getfile)
        
        self.btn1 = QPushButton("Save file")
        self.btn1.clicked.connect(self.savefile)

        # Just some button connected to `plot` method
        self.button = QPushButton('Recalculate/replot')
        self.button.clicked.connect(self.replot)

        #add text input boxes 
        self.textbox1 = QLineEdit('50')
        self.textbox2 = QLineEdit('2')
        self.textbox3 = QLineEdit('9')
        self.textbox4 = QLineEdit('2')
        
        #add text labels
        self.qlabel1 = QLabel('max eigenvectors')
        self.qlabel2 = QLabel('lowest eigenvalue')
        self.qlabel3 = QLabel('highest eigenvalue')
        self.qlabel4 = QLabel('window size (1/n)')
        
        # set the layout
        #layout = QHBoxLayout()
        layout = QGridLayout()
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, .1)
        
        
        #layout.addWidget(self.toolbar)
        #THESE ARE ZERO INDEXED...
        layout.addWidget(self.canvas,0,0,11,1)
        layout.addWidget(self.btn,0,1)
        layout.addWidget(self.btn1,1,1)
        layout.addWidget(self.button,2,1)
        layout.addWidget(self.qlabel1,3,1)
        layout.addWidget(self.textbox1,4,1)
        layout.addWidget(self.qlabel2,5,1)
        layout.addWidget(self.textbox2,6,1)
        layout.addWidget(self.qlabel3,7,1)
        layout.addWidget(self.textbox3,8,1)
        layout.addWidget(self.qlabel4,9,1)
        layout.addWidget(self.textbox4,10,1)
        
        
        #layout.addWidget(self.textbox1)
        self.setLayout(layout)
        
    def getfile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file') 
        self.data = detrend(np.loadtxt(str(fname)))
        self.maxvec =int(self.textbox1.text())
        winsize = int(self.textbox4.text())
        tsl=len(self.data)
        self.H=hankel(self.data[range(tsl/winsize)],self.data[range((tsl/winsize)-1,tsl)])
        self.U, self.s, self.V = randomized_svd(self.H, n_components=self.maxvec, n_iter=1, random_state=None)
        
    def savefile(self):
        sname = QFileDialog.getSaveFileName(self, 'Save file')
        np.savetxt(str(sname),self.svd_ts)
        
        #print str(self.filepath[0])

    def replot(self):
        #get data from textbox
        #textbox1Value = self.textbox1.text()
        textbox2Value = self.textbox2.text()
        textbox3Value = self.textbox3.text()
        #textbox4Value = self.textbox4.text()
        
        nmin=int(textbox2Value)
        nmax=int(textbox3Value)
        
        #winsize=2

        #H=hankel(drm.flatten()[range(tsl/winsize)],drm.flatten()[range((tsl/winsize)-1,tsl)])
        ##hankel([range(tsl/3)],[range((tsl/3)-1,tsl)])

        #maxvec=50
        #U, s, V = randomized_svd(H, n_components=maxvec, n_iter=2, random_state=None)



        #nrows,ncols=shape(H)
        #nmin=0
        #nmax=15
        S=np.zeros([self.maxvec,self.maxvec])
        for i in range(nmin,nmax):
            S[i,i]=self.s[i]
        ###reconstruct data set
        R=np.dot(self.U, np.dot(S, self.V))
        #i think this is correct, unless it introduces an off by one error
        self.svd_ts=np.hstack((R[:,0],R[-1,1:]))
        

        ##B=R[0].reshape(70,70)
        ##imshow(svd_ts[0:-1].reshape(shape(drm)));show()

        
        
        data=self.data
        ''' plot some random stuff '''
        # random data
        #data = [random.random() for i in range(10)]
        #data=np.loadtxt(self.filepath)

        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        #ax3 = self.figure.add_subplot(133)
        # plot data
        ax1.plot(data)
        ax1.plot(self.svd_ts)
        ax1.set_xlabel('index')
        ax1.set_ylabel('value')
        #ax1.text(.1,.1,textbox1Value)
        ax2.plot(np.log10(self.s))
        ax2.set_xlabel('eigenvalue index')
        ax2.set_ylabel('log10(eigenvalue)')
        self.figure.tight_layout()
        #ax3.plot(data)
        # refresh canvas
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Window()
    main.show()

    sys.exit(app.exec_())