#!/usr/bin/python3

import os,sys,time
from templates.Qt import QtWidgets,QtCore
import numpy as np

from templates import ui_layout as layout
from templates import ui_frameObject as object_layout
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QInputDialog, QLineEdit, QFileDialog

from datetime import datetime

TOTAL_STACK_FRAMES = 200
class AppWindow(QtWidgets.QMainWindow, layout.Ui_MainWindow):
	RESIZE= 128
	RESIZEFFT = 128
	ROIAveraging = 0.1
	FFTAveraging = 0.1
	frame_number=0
	centroid_number = 0
	centroid_list = np.empty(300)
	timestamps = np.empty(300)
	#voltstamps = np.empty(300)
	start_time = time.time()
	fps = None
	noisy_frames = None
	noisy_frames2 = None
	servoPosition = -300
	compare_image = None
	stack = None
	frames_done = 0
	advancedEnabled = True
	TOTAL_STACK_FRAMES = TOTAL_STACK_FRAMES
	KEITHLEY1_VALUE = 1
	KEITHLEY2_VALUE = 0
	KEITHLEY1_VALUE_STEPSIZE = 0.005 #10mV
	offlineMode = False
	def __init__(self, parent=None,**kwargs):
		super(AppWindow, self).__init__(parent)
		self.setupUi(self)

		self.record_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("r"), self)
		self.record_shortcut.activated.connect(self.record_fft)

		self.compare_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("s"), self)
		self.compare_shortcut.activated.connect(self.get_stack)

		self.setTheme("default")
		self.statusBar = self.statusBar()


		self.splash = kwargs.get('splash',None)
		self.cam = kwargs.get('cam',None)
		self.webcam = kwargs.get('webcam',None)
		self.vc = kwargs.get('voicecoil',None)
		self.servo = kwargs.get('servo',None)
		self.DAC = kwargs.get('DAC',None)
		self.KEITHLEY1 = kwargs.get('tmc_dac',None)
		self.KEITHLEY2 = kwargs.get('tmc_dac',None)
		self.KEITHLEY1_VALUE = 1
		if self.DAC is not None:
			self.DAC = self.DAC.MCP4725_set

		if self.cam: #XIMEA camera supplied. initialize it
			self.XIMEA_IMAGE = xiapi.Image()
			self.cam.start_acquisition()
		self.splash.showMessage("<h2><font color='Green'>Window ready...</font></h2>", QtCore.Qt.AlignLeft)
		self.splash.pbar.setValue(7)


		self.fpslabel = pg.LabelItem(justify='left')
		self.mainGraph.addItem(self.fpslabel)
		# Item for displaying RAW image data
		self.raw_view = self.mainGraph.addViewBox(row=1, col=0)
		self.raw_view.setAspectLocked(True)
		self.raw_img = pg.ImageItem(border='w')
		self.raw_view.addItem(self.raw_img)
		
		
		# Item for displaying Variation data
		self.var_view = self.lineGraph.plot()
		self.centroid_view = self.lineGraph.plot()
		self.centroid_view.setPen(color=(255,50,50))
		self.lineGraph.setRange(xRange=[-5, 0])


		self.mainBead = self.bead(raw_view = self.raw_view, raw_img=self.raw_img, status = self.statusBar,color=(0,255,0),title="Main Bead")
		self.beadLayout.addWidget(self.mainBead)
		self.referenceBead = self.bead(raw_view = self.raw_view, raw_img=self.raw_img, status = self.statusBar,color=(0,0,255),title="Reference Bead")
		self.beadLayout.addWidget(self.referenceBead)

		self.lastTime = ptime.time()

		self.timer = QtCore.QTimer()
		self.timer.timeout.connect(self.update)
		self.timer.start(0)

		self.stacktimer = QtCore.QTimer()
		self.stacktimer.timeout.connect(self.acquireStack)


		if self.vc and self.servo:
			self.servo.disable()
			self.servo.mode = il.SERVO_MODE.PP
			self.servo.enable()
			self.vc.start()
			self.timervc = QtCore.QTimer()
			self.timervc.timeout.connect(self.poll_vc)
			self.timervc.start(10)

	def showDock(self,state):
		self.advancedEnabled = state
		if state:
			self.dockWidget.show()
		else:
			self.dockWidget.close()
	def clearData(self):
		self.centroid_list = np.empty(300)
		self.timestamps = np.empty(300)
		self.start_time = time.time()
		self.centroid_number = 0
		now = datetime.now()
		start_time = now.strftime("%H:%M:%S")
		print("Clear Time =", start_time)

		
	class bead(QtWidgets.QWidget, object_layout.Ui_Form):
		RESIZE= 128
		RESIZEFFT = 128
		ROIAveraging = 0.1
		FFTAveraging = 0.1
		frame_number=0
		noisy_frames = None
		noisy_frames2 = None
		compare_image = None
		stack = None
		frames_done = 0
		TOTAL_STACK_FRAMES = TOTAL_STACK_FRAMES
		recording = False
		recording_update_time = time.time()


		def __init__(self, parent=None,**kwargs):
			super(AppWindow.bead, self).__init__(parent)
			self.setupUi(self)
			self.raw_view = kwargs.get('raw_view',None)
			self.raw_img = kwargs.get('raw_img',None)
			self.RESIZE = 128
			self.setWindowTitle(kwargs.get('title','Bead'))

			self.ref_roi = pg.RectROI([170, 270], [128, 128],True, pen=kwargs.get('color',(255,0,0)))
			self.ref_roi.aspectLocked=True

			self.raw_view.addItem(self.ref_roi)
			self.ref_roi.setZValue(10)  # make sure ROI is drawn above image


			# Item for displaying REFERENCE ROI image data
			self.roi_view = self.subGraph.addViewBox()
			self.roi_view.setAspectLocked(True)
			self.roi_img = pg.ImageItem(border='w')
			self.roi_view.addItem(self.roi_img)

			# Item for displaying ROI SPATIAL FFT image data
			self.fft_view = self.FFTGraph.addViewBox()
			self.fft_view.setAspectLocked(True)
			self.fft_img = pg.ImageItem(border='w')
			self.fft_view.addItem(self.fft_img)

			self.comp_roi = pg.RectROI([int(4.*self.RESIZE/10),int(4.*self.RESIZE/10)],[int(2.*self.RESIZE/10.),int(2.*self.RESIZE/10.)], True,pen=(0,9))
			self.comp_roi.aspectLocked=True

			self.fft_view.addItem(self.comp_roi)
			self.comp_roi.setZValue(10)  # make sure ROI is drawn above image
			'''
			# Item for displaying Variation data
			self.var_view = self.lineGraph.plot()
			self.centroid_view = self.lineGraph.plot()
			self.centroid_view.setPen(color=(255,50,50))
			'''

			# Item for displaying Fitted data
			self.extracted_view = self.fittedGraph.plot()
			self.fitted_view = self.fittedGraph.plot()
			self.fitted_view.setPen(color=(255,50,50))
		
		def initRecord(self,filename):
			self.out = open(filename,'wb')#cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'XVID'), 20, (int(height),int(width)) )
			pickle.dump(self.stack,self.out)
			pickle.dump(self.noisy_frames,self.out)
			pickle.dump(self.noisy_frames2,self.out)
			pickle.dump(self.compare_image,self.out)
			self.recording = True
			self.recording_update_time = 0

		def stopRecord(self):
			self.out.close()
			self.recording = False

		def setROIAveraging(self,val):
			self.ROIAveraging = val/100.
			self.ROIAveragingLabel.setText('K filter:%.2f'%(self.ROIAveraging))

		def setFFTAveraging(self,val):
			self.FFTAveraging = val/100.
			self.FFTAveragingLabel.setText('K filter:%.2f'%(self.FFTAveraging))

		def setAndProcess(self,frame,TIMESTAMP,updateGraphs = True,offlineMode = False):
			'''
			Takes a frame, and analyzes it with the following steps
			1) extracts bead area from the frame using the ref_roi rectangular region
			2) Applies Kalman(Moving average) filter if enabled
			3) Resizes ROI to RESIZE, RESIZE square.
			4) Takes spatial FFT, and plots the same
			5) If stack is available
				1) Compare with stack
				2) Gaussian FIT
				3) Return Centroid
			'''
			if offlineMode:
				self.selected = frame
			else:
				self.selected = self.ref_roi.getArrayRegion(frame, self.raw_img).astype(np.uint16)

			#print("image of size = " + str(self.selected.size) + " of shape " + str(self.selected.shape) + "...")
			
			if self.recording:
				self.roi_img.setImage(self.selected)#, autoLevels=False, lut=None, autoDownsample=False)
				pickle.dump([TIMESTAMP,self.selected],self.out)
				if TIMESTAMP - self.recording_update_time > 5:
					self.recording_update_time = TIMESTAMP
				else:
					return

			if self.ROIAveragingLabel.isChecked():
				if self.noisy_frames is None:
					self.noisy_frames = np.float64(self.selected)
				try:
					cv2.accumulateWeighted(self.selected,self.noisy_frames,self.ROIAveraging)
					self.selected = np.abs(self.noisy_frames)
				except:
					self.noisy_frames = np.float64(self.selected)

			self.selected = cv2.resize(self.selected,  (self.RESIZE,self.RESIZE),cv2.INTER_NEAREST)

			if self.ROIViewEnabled.isChecked() and updateGraphs:
				#self.selected = cv2.cvtColor(selected, cv2.COLOR_GRAY2RGB)
				self.roi_img.setImage(self.selected)#, autoLevels=False, lut=None, autoDownsample=False)


			################### CALCULATE FFT OF ROI ####################
			#print("Fourier transforming array of size = " + str(self.selected.size) + " of shape " + str(self.selected.shape) + "...")
			f = np.fft.fft2(self.selected)
			fshift = np.fft.fftshift(f)
			#print("done")

			#magnitude_spectrum = np.abs(fshift)#20*np.log(np.abs(fshift))
			self.magnitude_spectrum = 20*np.log(np.abs(fshift)) 
			#self.magnitude_spectrum = self.magnitude_spectrum[int(3.*self.RESIZE/10):int(7.*self.RESIZE/10),int(3.*self.RESIZE/10):int(7.*self.RESIZE/10.)]# cv2.resize(self.magnitude_spectrum[int(3.*self.RESIZE/10):int(7.*self.RESIZE/10),int(3.*self.RESIZE/10):int(7.*self.RESIZE/10.)],  (self.RESIZEFFT,self.RESIZEFFT),cv2.INTER_NEAREST)

			if self.FFTAveragingLabel.isChecked():
				if self.noisy_frames2 is None:
					self.noisy_frames2 = np.float64(self.magnitude_spectrum.astype(np.float32))
				try:
					cv2.accumulateWeighted(self.magnitude_spectrum.astype(np.float32),self.noisy_frames2,self.FFTAveraging)
					self.magnitude_spectrum = np.abs(self.noisy_frames2)
				except:
					self.noisy_frames2 = np.float64(self.magnitude_spectrum.astype(np.float32))

			if self.FFTViewEnabled.isChecked() and updateGraphs:
				self.fft_img.setImage(self.magnitude_spectrum.astype(np.float16))#, autoLevels=True, lut=None, autoDownsample=False)

			R = 4*self.RESIZE/10.
			CENTROID = None
			if self.compare_image is not None:
				#self.size_list.append(self.correlation_coefficient(self.compare_image,self.comp_roi.getArrayRegion(self.magnitude_spectrum, self.fft_img).astype(np.uint16))*100)
				CENTROID = self.compare_fft()
			else:
				CENTROID = np.average(self.magnitude_spectrum[int(R/2+5)][int(R/2)-2:int(R/2)+2])
				#self.size_list.append(CENTROID)


			return CENTROID

		def initAcquireStack(self):
			self.frames_done=0
			self.compare_image = self.comp_roi.getArrayRegion(self.magnitude_spectrum, self.fft_img).astype(np.uint16)
			width, height = self.compare_image.shape
			self.stack = np.zeros([self.TOTAL_STACK_FRAMES,width,height])

		def acquireStack(self):
			self.stack[self.frames_done] = self.comp_roi.getArrayRegion(self.magnitude_spectrum, self.fft_img).astype(np.uint16)
			self.frames_done +=1

		def compare_fft(self):
			if self.compare_image is not None:
				self.correlation_coefficient(self.compare_image,self.comp_roi.getArrayRegion(self.magnitude_spectrum, self.fft_img).astype(np.uint16))

				corr = []
				for i in self.stack:
					corr.append(self.correlation_coefficient(i,self.comp_roi.getArrayRegion(self.magnitude_spectrum, self.fft_img).astype(np.uint16)))

				if self.stack is None: return
				X = np.array(range(len(self.stack)))
				corr = np.array(corr)
				corr -= min(corr)
				self.extracted_view.setData(X,corr)
				try:
					centroid, X , corr = self.gaussianFit(X,corr)
					self.fitted_view.setData(X,corr)
					return centroid
				except Exception as error:
					print(error)
					return None

		def correlation_coefficient(self,patch1, patch2):
			product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
			stds = patch1.std() * patch2.std()
			if stds == 0:
				return 0
			else:
				product /= stds
				return product



		def gauss_erf(self,p,x,y):#p = [height, mean, sigma]
			return y - p[0] * np.exp(-(x-p[1])**2 /(2.0 * p[2]**2))

		def gauss_eval(self,x,p):
			return p[0] * np.exp(-(x-p[1])**2 /(2.0 * p[2]**2))


		def gaussianFit(self,X,Y):
			size = len(X)
			maxy = max(Y)
			halfmaxy = maxy / 2.0
			mean = sum(X*Y)/sum(Y)
			
			halfmaxima = X[int(len(X)/2)]
			for k in range(size):
				if abs(Y[k] - halfmaxy) < halfmaxy/10:
					halfmaxima = X[k]
					break
			sigma = mean - halfmaxima
			par = [maxy, mean, sigma] # Amplitude, mean, sigma				
			try:
				plsq = leastsq(self.gauss_erf, par,args=(X,Y))
			except:
				return None
			if plsq[1] > 4:
				print('fit failed')
				return None

			par = plsq[0]
			Xmore = np.linspace(X[0],X[-1],100)
			Y = self.gauss_eval(Xmore, par)

			return par[1],Xmore,Y

	def setKEITHLEY1(self,val1):
		if not self.KEITHLEY1: return
		#Write a voltage value in mV to the KEITHLEY 2230G sourcemeter.
		self.KEITHLEY1.write("INST:NSEL 1")
		self.KEITHLEY1.write("VOLT %.3f"%(val1/1000.))
		self.KEITHLEY1_VALUE = val1/1000. # V

	def setKEITHLEY2(self,val2):
		if not self.KEITHLEY2: return
		#Write a voltage value in mV to the KEITHLEY 2230G sourcemeter.
		self.KEITHLEY2.write("INST:NSEL 2")
		self.KEITHLEY2.write("VOLT %.3f"%(val2/1000.))
		self.KEITHLEY2_VALUE = val2/1000. # V	
		voltage = val2		

	def rampFunc(self,val3):
		if not self.KEITHLEY2: return
		self.KEITHLEY2.write("INST:NSEL 2")
		self.KEITHLEY2_VALUE = 170
		now = datetime.now()
		start_time = now.strftime("%H:%M:%S")
		print("Ramp start Time =", start_time)
		for val3 in range(170,500):
			self.KEITHLEY2.write("VOLT %.3f"%(val3/1000.))
			val3 = val3 + 1000
			time.sleep(0.01)
			QtGui.QApplication.processEvents()
		now = datetime.now()
		start_time = now.strftime("%H:%M:%S")
		print("Ramp end Time =", start_time)

	
	def get_stack(self):
		self.stacktimer.stop()
		time.sleep(0.5)
		self.mainBead.initAcquireStack()
		self.referenceBead.initAcquireStack()
		self.frames_done=0
		time.sleep(0.5) # settling delay.
		self.stacktimer.start(300)

	def saveStack(self):
		channels,width,height=self.stack.shape
		out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'XVID'), 20, (height,width) )
		for i in self.stack:
			out.write(cv2.cvtColor(np.uint8(i), cv2.COLOR_GRAY2RGB))
		out.release()

	def record_fft(self):
		self.compare_image = self.comp_roi.getArrayRegion(self.magnitude_spectrum, self.fft_img).astype(np.uint16)
		self.showStatus('Recorded')

	def setDac1(self,val1):
		self.DAC(val1)

	def setDac2(self,val2):
		self.DAC(val2)

	def setCurrent(self):
		val2 = self.magnetSlider.value()
		self.magnetPosition = val2
		#self.magnet.position = val2

	def saveData(self):
		np.savetxt('bead_data.txt',np.column_stack([self.timestamps[:self.centroid_number],self.centroid_list[:self.centroid_number]]))

	def setCurrentValue(self,val2):
		return
		val2 = self.magnetSlider.value()
		self.magnetPosition = -val2
		self.magnet.position = -val2

	def enableVoiceCoil(self,state):
		self.setKEITHLEY2(self.KEITHLEY2_VALUE*1000)
		if state: #retract
			self.KEITHLEY2_VALUE = 0
		else:  #re-deploy
			self.KEITHLEY2_VALUE = self.KEITHLEY2_VALUE

	def setExposure(self,val):
		self.cam.set_exposure(val)


	def poll_vc(self):
		try:
			t,d,s = self.vc.data
			pos = d[0][1]
			self.magnetLevel.setValue(pos)
			self.heightLabel.setText('%d'%pos)
		except:
			pass


	def recording(self,state):
		if state:
			from datetime import datetime
			now = datetime.now()
			start_time = now.strftime("%H:%M:%S")
			print("Start Time =", start_time)
			print('recording mode enabled')
			self.mainBead.initRecord('main.npy')
			self.referenceBead.initRecord('reference.npy')
			self.clearData()
			self.start_time = time.time()
			print("shape = ",comp_roi.shape)
		else:
			from datetime import datetime
			now = datetime.now()
			end_time = now.strftime("%H:%M:%S")
			print("End Time =", end_time)
			self.mainBead.stopRecord()
			self.referenceBead.stopRecord()
			print('recording mode stopped')
			self.clearData()
			self.start_time = time.time()
			
	def offlineAnalysis(self,state):
		if state:
			self.recordBox.setChecked(False)
			self.mainFile = open('main.npy','rb')#cv2.VideoCapture('main.avi')
			self.referenceFile = open('reference.npy','rb')#cv2.VideoCapture('reference.avi')
			self.offlineMode = True
			self.clearData()
			self.start_time = time.time()
			self.mainBead.stack = pickle.load(self.mainFile)
			self.mainBead.noisy_frames = pickle.load(self.mainFile)
			self.mainBead.noisy_frames2 = pickle.load(self.mainFile)
			self.mainBead.compare_image = pickle.load(self.mainFile)

			self.referenceBead.stack = pickle.load(self.referenceFile)
			self.referenceBead.noisy_frames = pickle.load(self.referenceFile)
			self.referenceBead.noisy_frames2 = pickle.load(self.referenceFile)
			self.referenceBead.compare_image = pickle.load(self.referenceFile)

		else:
			self.start_time = time.time()
			self.offlineMode = False
			self.clearData()
			

	def update(self):
		global ptr, lastTime, fps, img, data, cap,frames,areas,frame_number
		
		if self.offlineMode:
			try:
				TIMESTAMP,mainFrame = pickle.load(self.mainFile)
				if self.refReadBox.isChecked():
					TIMESTAMP,referenceFrame = pickle.load(self.referenceFile)
				#TIMESTAMP = time.time() - self.start_time
			except Exception as e:
				print('Reached end of file. ',e)
				np.savetxt('data.txt',np.column_stack([self.timestamps[:self.centroid_number],self.centroid_list[:self.centroid_number]]))
				pg.plot(self.timestamps[:self.centroid_number],self.centroid_list[:self.centroid_number])
				self.analyzeBox.setChecked(False)
				self.offlineAnalysis(False)
				self.clearData()
				return
		else:
			if self.cam: #Ximea camera was connected
				self.cam.get_image(self.XIMEA_IMAGE)
				frame = self.XIMEA_IMAGE.get_image_data_numpy()
				frame = cv2.flip(frame, 0)  # flip the frame vertically
				frame = cv2.flip(frame, 1)  # flip the frame horizontally
			elif self.webcam: #No ximea camera. Use webcam.
				_, frame = self.webcam.read()
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert to grayscale
			else:
				print('no camera')
				return
			TIMESTAMP = time.time() - self.start_time

		#print("Raw image of size " + str(frame.size) + " of shape = " + str(frame.shape))


		if self.rawEnabled.isChecked() and self.offlineMode == False: # SHOW THE CAMERA IMAGE `LIVE`.
			self.raw_img.setImage(frame)#, autoLevels=False, levels=None, lut=None, autoDownsample=False)

		#### PROCESSING AND PLOTTING THE ACQUIRED FRAME ###
		################### Extract reference from ROI ####################
		t = time.time()
		now = ptime.time()
		dt = now - self.lastTime
		self.lastTime = now
		#print("acquired raw image at time " + str(now) )
		if self.fps is None:
			self.fps = 1.0/dt
		else:
			s = np.clip(dt*3., 0, 1)
			self.fps = self.fps * (1-s) + (1.0/dt) * s


		if self.refReadBox.isChecked():
			if self.offlineMode:
				CA = self.mainBead.setAndProcess(mainFrame,TIMESTAMP,self.advancedEnabled,self.offlineMode)
				CB = self.referenceBead.setAndProcess(referenceFrame,TIMESTAMP,self.advancedEnabled,self.offlineMode)
			else:
				CA = self.mainBead.setAndProcess(frame,TIMESTAMP,self.advancedEnabled)
				CB = self.referenceBead.setAndProcess(frame,TIMESTAMP,self.advancedEnabled)
		else:
			if self.offlineMode:
				CA = self.mainBead.setAndProcess(mainFrame,TIMESTAMP,self.advancedEnabled,self.offlineMode)
				CB = 0
			else:
				CA = self.mainBead.setAndProcess(frame,TIMESTAMP,self.advancedEnabled)
				CB = 0

		if CA is not None and CB is not None:
			if self.refReadBox.isChecked():
				self.fpslabel.setText("<span style='font-size: 10pt;color:red;'>FPS=%4.1f</span>   |  %4.2f - %4.2f = %4.2f"%(self.fps,CA,CB,CA-CB))
				self.centroid_list[self.centroid_number] = CA
			else:
				self.fpslabel.setText("<span style='font-size: 10pt;color:red;'>FPS=%4.1f</span>   |  %4.2f"%(self.fps,CA))
				self.centroid_list[self.centroid_number] = CA
			self.timestamps[self.centroid_number] = TIMESTAMP
			self.centroid_number += 1

			## this works from 0 to 200 stacks without adding the drift correction check box ##
			## but causes a problem during stacking if correction window is narrowed down, say from stack 50 to 100 ##
			

			if self.centroid_number >= self.centroid_list.shape[0]-1:
				tmp = self.timestamps
				self.timestamps = np.empty(self.timestamps.shape[0] * 2) #double the size
				self.timestamps[:tmp.shape[0]] = tmp

				tmp = self.centroid_list
				self.centroid_list = np.empty(self.centroid_list.shape[0] * 2) #double the size
				self.centroid_list[:tmp.shape[0]] = tmp

		else:
			self.fpslabel.setText("<span style='font-size: 10pt;color:red;'>FPS=%4.1f</span>"%self.fps)
		self.frame_number+=1
		
		if (self.frames_done == self.TOTAL_STACK_FRAMES) and self.refReadBox.isChecked() and self.checkBox_3.isChecked():
			QtGui.QApplication.processEvents()
			if CB < 40:
				self.setKEITHLEY1(self.KEITHLEY1_VALUE*1000  + 2)
			elif CB > 60:
				self.setKEITHLEY1(self.KEITHLEY1_VALUE*1000 - 2)
		
		if self.centroid_number > 2:# and not self.mainBead.recording:
			self.centroid_view.setData(self.timestamps[:self.centroid_number],self.centroid_list[:self.centroid_number])
			self.centroid_view.setPos(-TIMESTAMP, 0)
		
	### drift correction function in AppWindow with check box added to the Main Window ###
	## This does not work, maybe bacuase it is not connected to the self updating function above ## 
	
	def driftCorr(self):				
		self.cam.get_image(self.XIMEA_IMAGE)			 
		frame = self.XIMEA_IMAGE.get_image_data_numpy()
		frame = cv2.flip(frame, 0)  # flip the frame vertically
		frame = cv2.flip(frame, 1)  # flip the frame horizontally
		TIMESTAMP = time.time() - self.start_time
		CB = self.referenceBead.setAndProcess(frame,TIMESTAMP,self.advancedEnabled)
		if self.checkBox_3.isChecked() and (self.frames_done == self.TOTAL_STACK_FRAMES):
			if CB < 50:
				self.setKEITHLEY1(self.KEITHLEY1_VALUE*1000  + 5)
			elif CB > 100:
				self.setKEITHLEY1(self.KEITHLEY1_VALUE*1000 - 5)
	

	def acquireStack(self):
		try:
			if self.frames_done < self.TOTAL_STACK_FRAMES:
				self.mainBead.acquireStack()
				self.referenceBead.acquireStack()
				self.frames_done+=1
				self.setKEITHLEY1(self.KEITHLEY1_VALUE*1000) #Volt to mV conversion
				self.KEITHLEY1_VALUE += self.KEITHLEY1_VALUE_STEPSIZE
				#MOVE THE DAC OUTPUT HERE
				self.showStatus("Acquiring Stack . Frames: %d/%d . V=%.3f"%(self.frames_done,self.TOTAL_STACK_FRAMES,self.KEITHLEY1_VALUE))
				if self.frames_done==self.TOTAL_STACK_FRAMES: #Complete
					self.showStatus("Finished acquiring stack",True)
					self.setKEITHLEY1((self.KEITHLEY1_VALUE - self.KEITHLEY1_VALUE_STEPSIZE*self.TOTAL_STACK_FRAMES/2)*1000) #Volt to mV conversion
				
		except Exception as e:
			print(e)
			self.frames_done = self.TOTAL_STACK_FRAMES
			self.showStatus("Failed to acquire stack",True)




	def setROIAveraging(self,val):
		self.ROIAveraging = val/100.
		self.ROIAveragingLabel.setText('K filter:%.2f'%(self.ROIAveraging))

	def setFFTAveraging(self,val):
		self.FFTAveraging = val/100.
		self.FFTAveragingLabel.setText('K filter:%.2f'%(self.FFTAveraging))
	def setTheme(self,theme):
		self.setStyleSheet("")
		self.setStyleSheet(open(os.path.join(path["themes"],theme+".qss"), "r").read())
	def showStatus(self,msg,error=None):
		if error: self.statusBar.setStyleSheet("color:#FAA")
		else: self.statusBar.setStyleSheet("color:#2F2")
		self.statusBar.showMessage(msg)
	def showMaxFrameRate(self):
		self.showStatus('Maximum FrameRate : ')




def translators(langDir, lang=None):
	"""
	create a list of translators
	@param langDir a path containing .qm translation
	@param lang the preferred locale, like en_IN.UTF-8, fr_FR.UTF-8, etc.
	@result a list of QtCore.QTranslator instances
	"""
	if lang==None:
			lang=QtCore.QLocale.system().name()
	result=[]
	qtTranslator=QtCore.QTranslator()
	qtTranslator.load("qt_" + lang,
			QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.TranslationsPath))
	result.append(qtTranslator)

	# path to the translation files (.qm files)
	sparkTranslator=QtCore.QTranslator()
	sparkTranslator.load(lang, langDir);
	result.append(sparkTranslator)
	return result

def firstExistingPath(l):
	"""
	Returns the first existing path taken from a list of
	possible paths.
	@param l a list of paths
	@return the first path which exists in the filesystem, or None
	"""
	for p in l:
		if os.path.exists(p):
			return p
	return None

def common_paths():
	"""
	Finds common paths
	@result a dictionary of common paths
	"""
	path={}
	curPath = os.path.dirname(os.path.realpath(__file__))
	path["current"] = curPath
	sharedPath = "/usr/share/beadtracker"
	path["translation"] = firstExistingPath(
			[os.path.join(p, "lang") for p in
			 (curPath, sharedPath,)])
	path["templates"] = firstExistingPath(
			[os.path.join(p,'templates') for p in
			 (curPath, sharedPath,)])

	path["splash"] = firstExistingPath(
			[os.path.join(p,'templates','splash.png') for p in
			 (curPath, sharedPath,)])
	path["themes"] = firstExistingPath(
			[os.path.join(p,'templates','themes') for p in
			 (curPath, sharedPath,)])

	lang=str(QtCore.QLocale.system().name()) 
	shortLang=lang[:2]
	path["help"] = firstExistingPath(
			[os.path.join(p,'HELP') for p in
			 (os.path.join(curPath,"help_"+lang),
			  os.path.join(sharedPath,"help_"+lang),
			  os.path.join(curPath,"help_"+shortLang),
			  os.path.join(sharedPath,"help_"+shortLang),
			  os.path.join(curPath,"help"),
			  os.path.join(sharedPath,"help"),
			  )
			 ])
	return path

def showSplash(pth = "splash"):
	path = common_paths()
	# Create and display the splash screen
	splash_pix = QtGui.QPixmap(path[pth])
	splash = QtWidgets.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
	splash.setMask(splash_pix.mask())

	progressBar = QtWidgets.QProgressBar(splash)
	progressBar.setStyleSheet('''

	QProgressBar {
		border: 2px solid grey;
		border-radius: 5px;
		border: 2px solid grey;
		border-radius: 5px;
		text-align: center;
	}
	QProgressBar::chunk {
		background-color: #012748;
		width: 10px;
		margin: 0.5px;
	}
	''')
	progressBar.setMaximum(10)
	progressBar.setGeometry(0, splash_pix.height() - 50, splash_pix.width(), 20)
	progressBar.setRange(0,8)

	splash.show()
	splash.pbar = progressBar
	splash.show()
	return splash


if __name__ == "__main__":
	path = common_paths()
	app = QtWidgets.QApplication(sys.argv)
	#for t in translators(path["translation"]):
	#	app.installTranslator(t)

	# Create and display the splash screen
	from templates.Qt import QtGui, QtCore, QtWidgets
	_translate = QtCore.QCoreApplication.translate
	splash = showSplash()
	splash.showMessage("<h2><font color='white'>Initializing...</font></h2>", QtCore.Qt.AlignLeft, QtCore.Qt.black)

	for a in range(5):
		app.processEvents()
		time.sleep(0.01)

	#IMPORT LIBRARIES
	splash.showMessage("<h2><font color='white'>Importing libraries...</font></h2>", QtCore.Qt.AlignLeft, QtCore.Qt.black)
	splash.pbar.setValue(1)
	import string,glob,functools

	splash.showMessage("<h2><font color='white'>Importing OpenCV library...</font></h2>", QtCore.Qt.AlignLeft, QtCore.Qt.black)
	import cv2

	splash.showMessage("<h2><font color='white'>Importing Numpy...</font></h2>", QtCore.Qt.AlignLeft, QtCore.Qt.black)
	splash.pbar.setValue(2)
	try:
		from scipy.optimize import leastsq
	except:
		splash.showMessage("<h2><font color='white'>Scipy module missing...</font></h2>", QtCore.Qt.AlignLeft, QtCore.Qt.black)

	splash.showMessage("<h2><font color='white'>Importing PyQtgraph...</font></h2>", QtCore.Qt.AlignLeft, QtCore.Qt.black)
	splash.pbar.setValue(5)
	import pyqtgraph as pg
	import pickle
	import pyqtgraph.ptime as ptime
	from pyqtgraph import exporters
	#pg.setConfigOptions(antialias=True, background='w',foreground='k')
	splash.showMessage("<h2><font color='white'>Importing Utilities...</font></h2>", QtCore.Qt.AlignLeft, QtCore.Qt.black)
	splash.pbar.setValue(5)
	
	try:
		splash.showMessage("<h2><font color='white'>Importing Ximea Library...</font></h2>", QtCore.Qt.AlignLeft, QtCore.Qt.black)
		splash.pbar.setValue(3)
		from ximea import xiapi
		splash.showMessage("<h2><font color='white'>Opening Ximea Camera...</font></h2>", QtCore.Qt.AlignLeft, QtCore.Qt.black)
		splash.pbar.setValue(4)
		cam = xiapi.Camera()
		cam.open_device()
		cam.set_exposure(10000)
		XIMEA_IMAGE = xiapi.Image()
		cam.set_imgdataformat('XI_MONO16')
		print (cam.get_framerate_maximum())
		cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
		webcam = None
	except:
		splash.showMessage("<h2><font color='red'>Failed to open Ximea Camera...Trying Webcam</font></h2>", QtCore.Qt.AlignLeft, QtCore.Qt.black)
		app.processEvents()
		time.sleep(0.5)
		cam = None
		try:
			webcam = cv2.VideoCapture(0)
			webcam.set(3,320)
			webcam.set(4,240)
		except:
			webcam = None
	'''
	try:
		import kuttyPy as kp
		kp.MCP4725_init()
	except:
		kp = None
		pass
	'''
	kp = None
	

	import usbtmc
	tmc_dac = usbtmc.Instrument(0x05e6, 0x2230)
	tmc_dac.write("INSTrument:COMBine:OFF")
	tmc_dac.write("SYST:REM")
	tmc_id = tmc_dac.ask("*IDN?")
	splash.showMessage("<h3><font color='white'>KEITHLEY DAC:%s</font></h3>"%tmc_id, QtCore.Qt.AlignLeft, QtCore.Qt.black)

	try:
		tmc_dac.write("INSTrument:SELect CH1")
		#tmc_dac.write("INSTrument:SELect CH2")
		tmc_dac.write("APPLY CH1,1.0V,0.1A")
		#tmc_dac.write("APPLY CH2,0.0V,0.0A")
		tmc_dac.write("OUTPUT ON")
		tmc_dac.write("SYST:BEEP")
		time.sleep(0.3)
	except:
		tmc_dac = None
		splash.showMessage("<h2><font color='red'>KEITHLEY DAC: NOT FOUND</font></h2>", QtCore.Qt.AlignLeft, QtCore.Qt.black)
		time.sleep(0.5)
	
	try:
		#tmc_dac.write("INSTrument:SELect CH1")
		tmc_dac.write("INSTrument:SELect CH2")
		#tmc_dac.write("APPLY CH1,0.0V,0.1A")
		tmc_dac.write("APPLY CH2,0.0V,1.0A")
		tmc_dac.write("OUTPUT ON")
		tmc_dac.write("SYST:BEEP")
		time.sleep(0.3)
	except:
		tmc_dac = None
		splash.showMessage("<h2><font color='red'>KEITHLEY DAC: NOT FOUND</font></h2>", QtCore.Qt.AlignLeft, QtCore.Qt.black)
		time.sleep(0.5)
	



	now = datetime.now()
	start_time = now.strftime("%H:%M:%S")
	print("Start Time =", start_time)

	#RUN APP
	splash.showMessage("<h2><font color='white'>Launching GUI...</font></h2>", QtCore.Qt.AlignLeft, QtCore.Qt.black)
	splash.pbar.setValue(6)
	if cam or webcam:
		myapp = AppWindow(app=app, path=path, splash=splash, cam = cam,webcam = webcam,DAC = kp, tmc_dac = tmc_dac)
		myapp.show()
		splash.finish(myapp)
		r = app.exec_()
		app.deleteLater()
		sys.exit(r)
	else:
		splash.finish(None)
		sys.exit(1)
