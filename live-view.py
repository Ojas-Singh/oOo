
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
import usbtmc
from numba import jit
def work(frame):
 
    f = np.fft.fft2(frame)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
    
    return magnitude_spectrum

if __name__ == '__main__':
    tmc_dac = usbtmc.Instrument(0x05e6, 0x2230)
    tmc_dac.write("INSTrument:COMBine:OFF")
    tmc_dac.write("SYST:REM")
    tmc_id = tmc_dac.ask("*IDN?")
    try:
        tmc_dac.write("INSTrument:SELect CH1")
        tmc_dac.write("INSTrument:SELect CH2")
        tmc_dac.write("APPLY CH1,1.0V,0.1A")
        tmc_dac.write("APPLY CH2,0.0V,0.0A")
        tmc_dac.write("OUTPUT ON")
        tmc_dac.write("SYST:BEEP")
    #    time.sleep(0.3)
    except:
        tmc_dac = None
        print("KEITHLEY DAC: NOT FOUND")
        time.sleep(0.5)

    cam = xiapi.Camera()
    print('Opening first camera...')
    cam.open_device()
    cam.set_exposure(1000)
    cam.set_param('width',256)
    cam.set_param('height',256)
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
        cv2.imshow("frame",frame)
        fps.update() 
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord('q'):
            break
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    cam.get_image(img)
    frame = img.get_image_data_numpy()
    frame = cv2.flip(frame, 0)  # flip the frame vertically
    frame = cv2.flip(frame, 1)
    roimain=cv2.selectROI("Main Bead select",frame)
    cv2.destroyAllWindows()
    roiref =cv2.selectROI("Main ref select",frame)
    cv2.destroyAllWindows()
    while True:
        cam.get_image(img)
        frame = img.get_image_data_numpy()
        frame = cv2.flip(frame, 0)  # flip the frame vertically
        frame = cv2.flip(frame, 1)
        fft = work(frame[int(roimain[1]):int(roimain[1]+roimain[3]), int(roimain[0]):int(roimain[0]+roimain[2])])  
        fft = cv2.resize(fft, (256, 256),interpolation = cv2.INTER_NEAREST)
        fft = cv2.rectangle(fft, (int(256*0.4),int(256*0.4)),(int(256*0.6),int(256*0.6)) , (255, 0, 0), 1)
        cv2.imshow("FFT",fft)
        cv2.imshow("LIVE",frame)
        fps.update() 
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord('q'):
            break
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    frame_count = 0
    stacksize = 200
    stack=[]
    stackref=[]
    KEITHLEY1_VALUE = 1
    KEITHLEY1_VALUE_STEPSIZE = 0.005 #10mV
    while frame_count < stacksize :
        
        tmc_dac.write("INST:NSEL 1")
        tmc_dac.write("VOLT %.3f"%(KEITHLEY1_VALUE))
        KEITHLEY1_VALUE += KEITHLEY1_VALUE_STEPSIZE
        frame_count +=1
        time.sleep(0.1)
        cam.get_image(img)
        frame = img.get_image_data_numpy()
        frame = cv2.flip(frame, 0)  # flip the frame vertically
        frame = cv2.flip(frame, 1)
        stack.append(work(frame[int(roimain[1]):int(roimain[1]+roimain[3]), int(roimain[0]):int(roimain[0]+roimain[2])]))
        cam.get_image(img)
        frame = img.get_image_data_numpy()
        frame = cv2.flip(frame, 0)  # flip the frame vertically
        frame = cv2.flip(frame, 1)
        stackref.append(work(frame[int(roiref[1]):int(roiref[1]+roiref[3]), int(roiref[0]):int(roiref[0]+roiref[2])]))
    print(len(stack),len(stackref))
    with open('stack.pkl', 'wb') as f:
        load=[stack,stackref,roimain,roiref]
        pickle.dump(load, f)
        f.close()
    tmc_dac.write("INST:NSEL 1")
    tmc_dac.write("VOLT %.3f"%((KEITHLEY1_VALUE - KEITHLEY1_VALUE_STEPSIZE*stacksize/2)))
    fps.stop()    
    
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cam.stop_acquisition()
    cam.close_device() 
    os._exit(1)
    # sys.exit()
    
    
    
    

    
    
