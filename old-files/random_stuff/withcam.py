from ximea import xiapi
from imutils.video import FPS
import imutils
import cv2
import numpy as np
cam = xiapi.Camera()
print('Opening first camera...')
cam.open_device()
cam.set_exposure(1000)
cam.set_param('width',128)
cam.set_param('height',128)
cam.set_param('downsampling_type', 'XI_SKIPPING')
cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FREE_RUN')

img = xiapi.Image()
print('Starting data acquisition...')
cam.start_acquisition()
fps = FPS().start()
i=0
buffer = []
pbuffer =[]
stack = []

while True:
    if i>5000:
        break
    cam.get_image(img)
    # data_raw = img.get_image_data_raw()
    frame = img.get_image_data_numpy()
    buffer.append(frame)
    if len(buffer) > 1000 :
        buffer.pop(0)
    Mathematics(frame,buffer,stack)
    # frame =cv2.flip(frame,0)
    # frame =cv2.flip(frame,1)
    # cv2.imshow("Frame", 20*frame)
    # cv2.waitKey(1)
    fps.update()
    i+=1

cam.stop_acquisition()
print(cam.get_param('framerate:max')) 
print(len(buffer))
cam.close_device()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()

###### All functions ####
def Mathematics(frame,buffer,stack):
    # if not GPU:
    frame = np.complex128(frame)
    f = np.fft.fft2(frame)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return 

def compare_fft():
    return

def correlation_coefficient():
    return
def gauss_erf(p,x,y):
    return y - p[0] * np.exp(-(x-p[1])**2 /(2.0 * p[2]**2))
def gauss_eval(x,p):
    return p[0] * np.exp(-(x-p[1])**2 /(2.0 * p[2]**2))
def gaussianFit(X,Y):
    return
