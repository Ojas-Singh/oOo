
from ximea import xiapi
import cv2
import numpy as np
import time
import sys,os
import pickle
from termcolor import colored
import config as config

def work(frame):
    f = np.fft.fft2(frame)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
    return magnitude_spectrum

if __name__ == '__main__':
    cam = xiapi.Camera()
    print('Opening first camera...')
    cam.open_device()
    cam.set_exposure(config.exposure)
    cam.set_param('width',config.resolution)
    cam.set_param('height',config.resolution)
    cam.set_param('downsampling_type', 'XI_SKIPPING')
    cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FREE_RUN')

    img = xiapi.Image()
    print('Starting data acquisition...')
    cam.start_acquisition()

    while True :
        cam.get_image(img)
        frame = img.get_image_data_numpy()
        frame = cv2.flip(frame, 0)  # flip the frame vertically
        frame = cv2.flip(frame, 1)
        cv2.imshow("Xemia Camera LIVE ",frame)
        fps.update() 
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord('q'):
            break
        cv2.waitKey(2)
    cv2.destroyAllWindows()
    cam.get_image(img)
    frame = img.get_image_data_numpy()
    frame = cv2.flip(frame, 0)  # flip the frame vertically
    frame = cv2.flip(frame, 1)
    roimain=cv2.selectROI("Main Bead select",frame)
    cv2.destroyAllWindows()
    roiref=cv2.selectROI("Reference Bead select",frame)
    cv2.destroyAllWindows()
    while True:
        cam.get_image(img)
        frame = img.get_image_data_numpy()
        frame = cv2.flip(frame, 0)  # flip the frame vertically
        frame = cv2.flip(frame, 1)
        fft = work(frame[int(roimain[1]):int(roimain[1]+roimain[3]), int(roimain[0]):int(roimain[0]+roimain[2])])  
        fft = cv2.resize(fft, (256, 256),interpolation = cv2.INTER_NEAREST)
        cv2.imshow("FFT",fft)
        cv2.imshow("LIVE",frame)
        fps.update() 
        key = cv2.waitKey(2) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord('q'):
            break
        cv2.waitKey(2)


    cam.stop_acquisition()
    cam.close_device() 
    os._exit(1)
    # sys.exit()
    
    
    
    

    
    
