from ximea import xiapi
from imutils.video import FPS
import cv2
import numpy as np
import time
import multiprocessing
from multiprocessing import Pool, Queue
import sys,os
import pickle
import random
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from numba import jit
matplotlib.use("Qt5agg")
from pipython import GCSDevice, pitools
from scipy import signal
import FKA
np.seterr(divide = 'ignore') 
np.seterr(over='ignore')
CONTROLLERNAME = 'E-709'
STAGES = None  
REFMODE = None

def roi128(j):
    x=int(j[0])+int(j[2]/2)
    y=int(j[1])+int(j[3]/2)
    return [x-64,y-64,128,128]



def zoom(f):
    return cv2.resize(f, (256, 256),
               interpolation = cv2.INTER_LINEAR)

def work(frame):
    window = signal.windows.kaiser(51, beta=14)
    b=np.kaiser(128,12)
    a=np.sqrt(np.outer(b,b)) 
    frame=a*frame
    f = np.fft.fft2(frame)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 200*np.log10(np.abs(fshift))
    magnitude_spectrum = np.asarray(magnitude_spectrum)
    # print(magnitude_spectrum)
    return magnitude_spectrum,frame,a

def avgframe(list):
    rAvg = None
    total = 0
    for frame in list:
        if rAvg is None:
            rAvg = frame
        # otherwise, compute the weighted average between the history of
        # frames and the current frames
        else:
            rAvg = ((total * rAvg) + (1 * frame)) / (total + 1.0)

    avg = rAvg
    return avg 


if __name__ == '__main__':
    with GCSDevice(CONTROLLERNAME) as pidevice:
        pidevice.InterfaceSetupDlg(key='sample')
        # pidevice.ConnectUSB(serialnum='123456789')
        print('connected: {}'.format(pidevice.qIDN().strip()))
        if pidevice.HasqVER():
            print('version info: {}'.format(pidevice.qVER().strip()))
        pitools.startup(pidevice, stages=STAGES)
        pidevice.MOV(pidevice.axes, 2.00)
        pitools.waitontarget(pidevice)
        cam = xiapi.Camera()
        print('Opening first camera...')
        cam.open_device()
        cam.set_exposure(1500)
        cam.set_param('width',1280)
        cam.set_param('height',1024)
        cam.set_param('downsampling_type', 'XI_SKIPPING')
        cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FREE_RUN')

        
        img = xiapi.Image()
        print('Starting data acquisition...')
        cam.start_acquisition()
        fps = FPS().start()

        
        while True :
            cam.get_image(img)
            frame = img.get_image_data_numpy()
            frame = cv2.flip(frame, 0)  # flip the frame vertically
            frame = cv2.flip(frame, 1)
            # frame =  cv2.rectangle(frame, (512,512),(1024,1024) , (255, 0, 0), 1)
            cv2.imshow("frame",frame)
            fps.update() 
            key = cv2.waitKey(2) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord('q'):
                break
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        # cam.set_param('width',512)
        # cam.set_param('height',512)
        cam.get_image(img)
        frame = img.get_image_data_numpy()
        frame = cv2.flip(frame, 0)  # flip the frame vertically
        frame = cv2.flip(frame, 1)
        roimain= []
        val1 = int(input("Enter the number of Magnetic Beads to process : "))
        for i in range(val1):
            k=1
            if len(roimain)>0 :
                for j in roimain:
                    frame = cv2.rectangle(frame, (int(j[0]), int(j[1])), (int(j[0])+int(j[2]), int(j[1])+int(j[3])), (36,255,12), 1)
                    cv2.putText(frame, "Bead "+str(k), (int(j[0]), int(j[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    k+=1
            roi = cv2.selectROI("Main Bead"+str(i+1)+" select",frame)
            print(roi)
            roimain.append(roi128(roi))
            cv2.destroyAllWindows()
            
        k=1
        if len(roimain)>0 :
            for j in roimain:
                frame = cv2.rectangle(frame, (int(j[0]), int(j[1])), (int(j[0])+int(j[2]), int(j[1])+int(j[3])), (36,255,12), 1)
                cv2.putText(frame, "Bead "+str(k), (int(j[0]), int(j[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                k+=1
        roiref =cv2.selectROI("Main ref select",frame)
        roiref = roi128(roiref)
        cv2.destroyAllWindows()
        # roiref =cv2.selectROI("Main ref select",frame)
        cv2.destroyAllWindows()
        photo= False
        while True:
            cam.get_image(img)
            frame = img.get_image_data_numpy()
            frame = cv2.flip(frame, 0)  # flip the frame vertically
            frame = cv2.flip(frame, 1)
            k=1
            if len(roimain)>0 :
                for j in roimain:
                    frame = cv2.rectangle(frame, (int(j[0]), int(j[1])), (int(j[0])+int(j[2]), int(j[1])+int(j[3])), (36,255,12), 1)
                    cv2.putText(frame, "Bead "+str(k), (int(j[0]), int(j[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    k+=1
                frame = cv2.rectangle(frame, (int(roiref[0]), int(roiref[1])), (int(roiref[0])+int(roiref[2]), int(roiref[1])+int(roiref[3])), (36,255,12), 1)
                cv2.putText(frame, "Ref Bead ", (int(roiref[0]), int(roiref[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            fft = work(frame[int(roimain[0][1]):int(roimain[0][1]+roimain[0][3]), int(roimain[0][0]):int(roimain[0][0]+roimain[0][2])])[0]
            fft = cv2.resize(fft, (256, 256),interpolation = cv2.INTER_NEAREST)
            
            cv2.imshow("FFT of Bead 1",np.asarray(fft, dtype=np.uint8))
            cv2.imshow("LIVE",frame)
            if not photo :
                cv2.imwrite("output/image.jpg", frame)
                photo= True
            fps.update() 
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord('q'):
                break
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        # cam.set_param('imgdataformat', 'XI_RAW16')
        frame_count = 0
        stacksize = 200
        stack=[]
        stackref=[]
        imga=[]
        kai=[]
        k_imga=[]
        wok=[]
        VALUE = 2.0
        VALUE_STEPSIZE = 0.01 
        pidevice.MOV(pidevice.axes, VALUE)
        pitools.waitontarget(pidevice)
        stackrange=[]
        while frame_count < stacksize :
            pidevice.MOV(pidevice.axes, VALUE)
            pitools.waitontarget(pidevice)
            print(pidevice.qPOS(pidevice.axes))
            stackrange.append(float(pidevice.qPOS(pidevice.axes)['Z']))
            VALUE += VALUE_STEPSIZE
            frame_count +=1
            
            stackm=[]
            for roim in roimain:
                lis = []
                for i in range(10):
                    cv2.destroyAllWindows()
                    cam.get_image(img)
                    frame = img.get_image_data_numpy()
                    # print(frame)
                    frame = cv2.flip(frame, 0)  # flip the frame vertically
                    frame = cv2.flip(frame, 1)
                    lis.append(frame[int(roim[1]):int(roim[1]+roim[3]), int(roim[0]):int(roim[0]+roim[2])])
                    
                    
                avg = avgframe(lis)
                imga.append(avg)
                cv2.imshow("FFT of Bead 1",(zoom(np.asarray(work(avg)[0], dtype=np.uint8))))
                cv2.waitKey(5)
                # cv2.imshow("FFT of Bead 1",zoom(work(avg)))
                ll=zoom(work(avg)[0])
                l2=ll.tolist()
                stackm.append(FKA.FKA(l2))
                k_imga.append(work(avg)[1])
                kai.append(work(avg)[2])
                wok.append(work(avg)[0])
                # stackm.append(zoom(work(avg)))
            stack.append(stackm)

            cam.get_image(img)
            frame = img.get_image_data_numpy()
            frame = cv2.flip(frame, 0)  # flip the frame vertically
            frame = cv2.flip(frame, 1)
            l3=zoom(work(frame[int(roiref[1]):int(roiref[1]+roiref[3]), int(roiref[0]):int(roiref[0]+roiref[2])])[0])
            l4= l3.tolist()
            stackref.append(FKA.FKA(l4))
        print(len(stack),len(stackref))
        # print(stackrange)
        with open('stack.pkl', 'wb') as f:
            load=[stack,stackref,roimain,roiref,stackrange,imga,kai,k_imga,wok]
            pickle.dump(load, f)
            f.close()
        pidevice.MOV(pidevice.axes,3.00)
        pitools.waitontarget(pidevice)
        print("Final position :",pidevice.qPOS(pidevice.axes))
        fps.stop()    
        
        print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
        print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

        cam.stop_acquisition()
        cam.close_device() 
        os._exit(1)
        
        
        
        

        
        
