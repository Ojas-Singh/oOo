from ximea import xiapi
from imutils.video import FPS
import imutils
import cv2
import numpy as np
from queue import Queue
from threading import Thread
import time


RESIZE = 50000
def worker(input_q, output_q):
    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
        ### 
        
        output = imutils.resize(frame, width=RESIZE, height=RESIZE)
        output_q.put(output)

    fps.stop()


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


if __name__ == '__main__':
    qu_limit = 1000
    threadn = cv2.getNumberOfCPUs() -2
    print("Threads : ", threadn)
    input_q = Queue(qu_limit)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    for i in range(threadn):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

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
    frame_count = 0
    while True and frame_count< 10000:
        frame_count += 1
        cam.get_image(img)
        # data_raw = img.get_image_data_raw()
        frame = 20*img.get_image_data_numpy()        
        # frame = video_capture.read()        
        if frame_count % qu_limit == 0:            
            input_q.put(frame)        

        if output_q.empty():
            pass  # fill up queue
        # else:                       
            # cv2.imshow('Video', frame)
        # if frame_count % 500 ==0 :
        #     print('[INFO] Live . FPS: {:.2f}'.format(fps.fps()))
        fps.update()        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    fps.stop()    
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
    cam.stop_acquisition()
    print("Max camera framerate :",cam.get_param('framerate:max')) 
    cam.close_device()
    cv2.destroyAllWindows()
