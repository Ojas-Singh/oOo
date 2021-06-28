
from ximea import xiapi
import cv2
import numpy as np
import time
import sys,os
import pickle
import random
from numba import jit
import usbtmc


def work(frame):
 
    f = np.fft.fft2(frame)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
    
    return magnitude_spectrum
           
def setKEITHLEY1(val1):
        KEITHLEY1 = tmc_dac
        if not KEITHLEY1: return
        #Write a voltage value in mV to the KEITHLEY 2230G sourcemeter.
        KEITHLEY1.write("INST:NSEL 1")
        KEITHLEY1.write("VOLT %.3f"%(val1/1000.))
        KEITHLEY1_VALUE = val1/1000. # V     

def aquirestack(frame,roimain,roiref,stacksize,frame_count):
    
    KEITHLEY1_VALUE = 1
    KEITHLEY1_VALUE_STEPSIZE = 0.005 #10mV
    if frame_count < stacksize:
        stack.append(frame[int(roimain[1]):int(roimain[1]+roimain[3]), int(roimain[0]):int(roimain[0]+roimain[2])])
        stackref.append(frame[int(roiref[1]):int(roiref[1]+roiref[3]), int(roiref[0]):int(roiref[0]+roiref[2])])
        setKEITHLEY1(KEITHLEY1_VALUE*1000) #Volt to mV conversion
        KEITHLEY1_VALUE += KEITHLEY1_VALUE_STEPSIZE
        frame_count +=1
    else:
        setKEITHLEY1((KEITHLEY1_VALUE - KEITHLEY1_VALUE_STEPSIZE*stacksize/2)*1000) #Volt to mV conversion
    



if __name__ == '__main__':
    
    tmc_dac = usbtmc.Instrument(0x05e6, 0x2230)
    tmc_dac.write("INSTrument:COMBine:OFF")
    tmc_dac.write("SYST:REM")
    tmc_id = tmc_dac.ask("*IDN?")
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
        print("KEITHLEY DAC: NOT FOUND")
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
        print("KEITHLEY DAC: NOT FOUND")
        time.sleep(0.5)
    cam = xiapi.Camera()
    print('Opening first camera...')
    cam.open_device()
    cam.set_exposure(2000)
    cam.set_param('width',512)
    cam.set_param('height',512)
    cam.set_param('downsampling_type', 'XI_SKIPPING')
    cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FREE_RUN')
 
    frame_count = 0
    stacksize = 200
    stack=[]
    stackref=[]

    
    
    img = xiapi.Image()
    print('Starting data acquisition...')
    cam.start_acquisition()
    cam.get_image(img)
    frame = img.get_image_data_numpy()
    frame = cv2.flip(frame, 0)  # flip the frame vertically
    frame = cv2.flip(frame, 1)
    roimain=cv2.selectROI("Main Bead select",frame)
    roiref =cv2.selectROI("Main ref select",frame)
    cv2.destroyAllWindows()
    print(roimain,roiref)


    while frame_count == stacksize :
        cam.get_image(img)
        aquirestack(img.get_image_data_numpy(),roimain,roiref,stacksize,frame_count)

        


    cam.stop_acquisition()
    cam.close_device() 
    os._exit(1)
    # sys.exit()
    
    
    
    

    
    
