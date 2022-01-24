from threading import stack_size
from ximea import xiapi
from imutils.video import FPS
import cv2
import numpy as np
import time
import multiprocessing
from multiprocessing import Pool, Queue
import sys,os
import pickle
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from numba import jit
matplotlib.use("Qt5agg")

@jit(nopython=True)
def correlation_coefficient( patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product
@jit(nopython=True)
def gauss_erf(p,x,y):#p = [height, mean, sigma]
	return y - p[0] * np.exp(-(x-p[1])**2 /(2.0 * p[2]**2))
@jit(nopython=True)
def gauss_eval(x,p):
	return p[0] * np.exp(-(x-p[1])**2 /(2.0 * p[2]**2))


def gaussianFit(X,Y):
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
		plsq = leastsq(gauss_erf, par,args=(X,Y))
	except:
		return None
	if plsq[1] > 4:
		print('fit failed')
		return None

	par = plsq[0]
	Xmore = np.linspace(X[0],X[-1],100)
	Y = gauss_eval(Xmore, par)

	return par[1],Xmore,Y

def worker(input_q, output_q,stack):
    RESIZE = 128
    while True:
        frameinfo = input_q.get() 


        f = np.fft.fft2(frameinfo[1])
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
        centroid = None
        R = 4 * RESIZE / 10
        corr = []

        for img in stack:
            # corr.append(correlation_coefficient(img, comp_roi.getArrayRegion(magnitude_spectrum)))
            corr.append(correlation_coefficient(img, magnitude_spectrum))

        X = np.array(range(len(stack)))
        corr = np.array(corr)
        corr -= min(corr)
        #self.extracted_view.setData(X, corr)
        try:
            centroid, X, corr = gaussianFit(X, corr)
            #self.fitted_view.setData(X, corr)
            output_q.put([frameinfo[0],centroid])
        except Exception as error:
            print(error)
            
        

def graphdisplayworker(graph_q):
    fig = plt.figure()
    data = [[],[]]
    ax = fig.add_subplot(111)
    fig.show()
    timestart = time.time()
    while True:
        
        if quit:
            break
        for j in range(graph_q.qsize()):
            timestamp,centroid = graph_q.get()
            data[0].append(timestamp-timestart)
            data[1].append(centroid)
        timenowplot = time.time()
        ax.plot(data[0], data[1], color='b')
        plt.pause(0.02)
        ax.set_xlim(left=max(0, timenowplot-timestart-3), right=timenowplot-timestart+1)
        # plt.pause(0.05)
        plt.show(block=False)
        time.sleep(.005)
        cv2.waitKey(1)
        
def record(display_q):
    results = []
    quit_state = False
    while not quit_state:
        data = display_q.get()
        timestamp,centroid = data[1]
        results.append((timestamp,centroid))
        graph_q.put((timestamp,centroid))
        quit_state = data[0]
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("written to file results.pkl !")


if __name__ == '__main__':
    cam = xiapi.Camera()
    print('Opening first camera...')
    cam.open_device()
    cam.set_exposure(1000)
    cam.set_param('width',128)
    cam.set_param('height',128)
    cam.set_param('downsampling_type', 'XI_SKIPPING')
    cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FREE_RUN')
    qu_limit = 10
    workers = 12
    threadn = cv2.getNumberOfCPUs() 
    print("Threads : ", threadn)
    print("Workers Spawned : ", workers)
    input_q = Queue(qu_limit)  # fps is better if queue is higher but then more lags
    frame_count = 0
    stacksize = 200
    stack=[]
    output_q = Queue()
    display_q = Queue()
    graph_q = Queue()
    quit = False
    all_processes = []
    
    D = multiprocessing.Process(target=graphdisplayworker, args=[graph_q],daemon = False)
    R = multiprocessing.Process(target=record, args=[display_q],daemon = False)
    

    
    
    img = xiapi.Image()
    print('Starting data acquisition...')
    cam.start_acquisition()
    fps = FPS().start()
    cam.get_image(img)
    frame = 20*img.get_image_data_numpy()
    roi=cv2.selectROI(frame)
    cv2.destroyAllWindows()
    for i in range(stacksize):
        cam.get_image(img)
        frame = 20*img.get_image_data_numpy()
        stack.append(frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])])
        cv2.waitKey(1)
    for i in range(workers):
        p = multiprocessing.Process(target=worker, args=[input_q, output_q,stack],daemon = True)
        p.start()
        all_processes.append(p)
    cv2.waitKey(2)
    R.start()
    D.start()
    
    while quit == False and frame_count <500:
        cam.get_image(img)
        frame = 20*img.get_image_data_numpy()
        input_q.put([time.time(),frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]])
        
    
        if output_q.empty():
            pass  # fill up queue
        else:
            frame_count += 1
            dummylist=[]
            for i in range(output_q.qsize()):
                dummylist.append((quit,output_q.get()))
            dummylist.sort()
            for i in dummylist:
                display_q.put(i)
            fps.update() 
        
                
    fps.stop()    
    quit = True
    
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
    display_q.put((quit,output_q.get()))
    time.sleep(4)
    D.terminate()
    R.terminate()

    for process in all_processes:
        process.terminate()
    cam.stop_acquisition()
    cam.close_device() 
    os._exit(1)
    # sys.exit()
    
    
    
    

    
    
