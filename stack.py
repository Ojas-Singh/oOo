
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
           
        
def aquirestack(roi,stacksize):
    return stack



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


    # for i in range(stacksize):
    #     cam.get_image(img)
    #     frame = 20*img.get_image_data_numpy()
    #     stack.append(frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])])
    #     cv2.waitKey(1)

        


    cam.stop_acquisition()
    cam.close_device() 
    os._exit(1)
    # sys.exit()
    
    
    
    

    
    
