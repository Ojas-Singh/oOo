from ximea import xiapi
from imutils.video import FPS, fps
import cv2
import numpy as np
import time
import multiprocessing
from multiprocessing import Queue, Value
import sys,os
import pickle
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from numba import jit
matplotlib.use("Qt5agg")
from termcolor import colored
import config 
import FKA
np.seterr(divide = 'ignore') 
np.seterr(over='ignore')
sys.tracebacklimit=0


def zoom(f):
    return cv2.resize(f, (256, 256),
               interpolation = cv2.INTER_LINEAR)

# @jit(nopython=True)
def correlation_coefficient( a, b):
    patch1=np.asarray(a)
    patch2=np.asarray(b)
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
    yne=[]
    yne2=[]
    Xn=[]
    Yn=[]
    Xp=[]
    siz= len(X)
    t=0
    t2=30
    for i in range(siz):
        yne.append((Y[i]**5,X[i]))
    yne=sorted(yne,reverse=True)
    for i in range(t,t2):
        yne2.append((yne[i][1],yne[i][0]))
        Xp.append(yne[i][1])
    # print(yne2[0])
    yne2= sorted(yne2)
    Xpp= np.asarray(Xp).mean()
    for i in yne2:
        if np.abs(i[0]-Xpp)< 300.0:
            Yn.append(i[1])
            Xn.append(i[0])
    
    # for i in range(t2):
    #     Yn.append(yne2[i][1])
    #     Xn.append(yne2[i][0])
    Xn= np.asarray(Xn)

    Yn= np.asarray(Yn)
    maxy = max(Yn)
    # print("MAX :",yne[0][1],"Corr :",Xn)
    size = len(Xn)
    halfmaxy = maxy / 2.0
    mean = sum(Xn*Yn)/sum(Yn)

    halfmaxima = Xn[int(len(Xn)/2)]
    for k in range(size):
        if abs(Yn[k] - halfmaxy) < halfmaxy/10:
            halfmaxima = Xn[k]
            break
    sigma = mean - halfmaxima
    par = [maxy, mean, sigma] # Amplitude, mean, sigma	
    #print(maxy)
    try:
        plsq = leastsq(gauss_erf, par,args=(Xn,Yn))
    except:
        return None
    # if maxy<0.5:
    #     return None
    if plsq[1] > 4:
        return None

    par = plsq[0]
    return par[1] 



def worker(input_q, output_q,stack):
    while True:
        frameinfo = input_q.get() 
        if frameinfo[1] is not None :
            b=np.kaiser(128,12)
            a=np.sqrt(np.outer(b,b)) 
            frame=frameinfo[1]*a
            f = np.fft.fft2(frame)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 200*np.log10(np.abs(fshift))
            magnitude_spectrum = np.asarray(magnitude_spectrum)
            magnitude0 = zoom(magnitude_spectrum)
            magnitude1=magnitude0.tolist()
            magnitude= FKA.FKA(magnitude1)
            mag= np.asarray(magnitude)
            centroid = None
            corr = []
            for im in stack:
                img= im[frameinfo[2]]
                # corr.append(correlation_coefficient(img[int(l1*0.45):int(l1*0.55),int(l2*0.45):int(l2*0.55)], magnitude_spectrum[int(l1*0.45):int(l1*0.55),int(l2*0.45):int(l2*0.55)]))
                # corr.append(correlation_coefficient(img[int(int(128-a)):int(int(128+a)),int(int(128-a)):int(int(128+a))], magnitude_spectrum[int(int(128-a)):int(int(128+a)),int(int(128-a)):int(int(128+a))]))
                corr.append(correlation_coefficient(img, mag))
                # corr.append(correlation_coefficient(img[int(l1*(0.5-a/2)):int(l1*(0.5+a/2)),int(l2*(0.5-a/2)):int(l2*(0.5+a/2))], magnitude_spectrum[int(l1*(0.5-a/2)):int(l1*(0.5+a/2)),int(l2*(0.5-a/2)):int(l2*(0.5+a/2))]))
            X= np.array([i*10 for i in range(len(stack))])
            # X = np.array(range(len(stack)))
            corr = np.array(corr)
            # corr -= min(corr)
            try:
                centroid = gaussianFit(X, corr)
                output_q.put([frameinfo[0],centroid,frameinfo[2]])
            except Exception as error:
                pass

def driftworker(drift_q, driftoutput_q,stackref):
    while True:
        frameinfo = drift_q.get() 
        if frameinfo[1] is not None :
            b=np.kaiser(128,12)
            a=np.sqrt(np.outer(b,b)) 
            frame=frameinfo[1]*a
            f = np.fft.fft2(frame)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20*np.log10(np.abs(fshift))
            magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
            magnitude0 = zoom(magnitude_spectrum)
            magnitude1=magnitude0.tolist()
            magnitude= FKA.FKA(magnitude1)
            mag= np.asarray(magnitude)
            centroid = None
            # print("Drift woerker")
            corr = []
            for img in stackref:
                corr.append(correlation_coefficient(img, mag))
                # corr.append(correlation_coefficient(img[int(int(128-a)):int(int(128+a)),int(int(128-a)):int(int(128+a))], magnitude_spectrum[int(int(128-a)):int(int(128+a)),int(int(128-a)):int(int(128+a))]))
            X= np.array([i*10 for i in range(len(stackref))])
            # X = np.array(range(len(stackref)))
            corr = np.array(corr)
            corr -= min(corr)
            try:
                centroid = gaussianFit(X, corr)
                # print(centroid)
                driftoutput_q.put((frameinfo[0],centroid))
            except Exception as error:
                pass             
def graphdisplayworker(graph_q):
    fig = plt.figure(figsize=(16, 3))
    data = [[],[]]
    ax = fig.add_subplot(111)
    fig.show()
    timestart = time.time()
    while True:
        for j in range(graph_q.qsize()):
            timestamp,centroid = graph_q.get()
            data[0].append(timestamp-timestart)
            data[1].append(centroid)
        timenowplot = time.time()
        ax.plot(data[0], data[1], color='b' ,linewidth=.2)
        plt.pause(0.03)
        ax.set_xlim(left=max(0, timenowplot-timestart-4), right=timenowplot-timestart+1.25)
        plt.show(block=False)
        time.sleep(.1)
        if (timenowplot - timestart ) > 3:
            data = [[],[]]
       
def record(display_q,driftoutput_q,graph_q,k):
    results = [[] for i in range(k)]
    driftrec =[]
    timestart = time.time()
    try:
        while True:
            
            drec = driftoutput_q.get()
            driftrec.append(drec)
            n=display_q.qsize()
            for j in range(n):
                data = display_q.get()
                timestamp,centroid,i = data[1]
                results[i].append((timestamp,centroid))
                if i==0:
                    graph_q.put((timestamp,centroid))
    except KeyboardInterrupt:
        tt = open("output/drift.txt", "w")
        for i in driftrec:
            tt.write(str(i[0]-timestart)+" "+str(i[1])+"\n")
        tt.close()
        print("written to file drift.txt !")
        for j in range(k):
            f=open("output/"+str(j+1)+".txt", "w")
            for i in results[j]:
                f.write(str(i[0]-timestart)+" "+str(i[1])+"\n")
            f.close()

            

  

if __name__ == '__main__':
    CONTROLLERNAME = 'E-709'
    STAGES = None  
    REFMODE = None

    print(colored(" ", 'yellow')    )                                                
    print(colored("  ", 'yellow')     )                                              
    print(colored("                      OOOOOOOOO                      ", 'yellow'))
    print(colored("                    OO:::::::::OO                    ", 'yellow'))
    print(colored("                  OO:::::::::::::OO                  ", 'yellow'))
    print(colored("                 O:::::::OOO:::::::O                 ", 'yellow'))
    print(colored("   ooooooooooo   O::::::O   O::::::O   ooooooooooo   ", 'yellow'))
    print(colored(" oo:::::::::::oo O:::::O     O:::::O oo:::::::::::oo ", 'yellow'))
    print(colored("o:::::::::::::::oO:::::O     O:::::Oo:::::::::::::::o", 'yellow'))
    print(colored("o:::::ooooo:::::oO:::::O     O:::::Oo:::::ooooo:::::o", 'yellow'))
    print(colored("o::::o     o::::oO:::::O     O:::::Oo::::o     o::::o", 'yellow'))
    print(colored("o::::o     o::::oO:::::O     O:::::Oo::::o     o::::o", 'yellow'))
    print(colored("o::::o     o::::oO:::::O     O:::::Oo::::o     o::::o", 'yellow'))
    print(colored("o::::o     o::::oO::::::O   O::::::Oo::::o     o::::o", 'yellow'))
    print(colored("o:::::ooooo:::::oO:::::::OOO:::::::Oo:::::ooooo:::::o", 'yellow'))
    print(colored("o:::::::::::::::o OO:::::::::::::OO o:::::::::::::::o", 'yellow'))
    print(colored(" oo:::::::::::oo    OO:::::::::OO    oo:::::::::::oo ", 'yellow'))
    print(colored("   ooooooooooo        OOOOOOOOO        ooooooooooo   ", 'yellow'))
    print(colored("                                                     ", 'yellow'))    
    print(colored(" ", 'yellow') )



    stackflag = True
    
    cam = xiapi.Camera()
    print('Opening first camera...')
    cam.open_device()
    cam.set_exposure(1500)
    # cam.set_param('imgdataformat', 'XI_RAW16')
    cam.set_param('width',1280)
    cam.set_param('height',1024)
    cam.set_param('downsampling_type', 'XI_SKIPPING')
    cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FREE_RUN')
    qu_limit = config.qu_limit
    workers = config.workers
    driftworkers=config.driftworker
    threadn = cv2.getNumberOfCPUs() 
    print("Threads : ", threadn)
    print("Workers Spawned : ", workers)
    input_q = Queue(qu_limit)  # fps is better if queue is higher but then more lags
    driftoutput_q = Queue() 
    frame_count = 0
    stack=[]
    stackref=[]
    roiref = None
    roimain = None
    output_q = Queue()
    display_q = Queue()
    graph_q = Queue()
    drift_q = Queue()
    drift_data = Queue()
    # p_output, p_input = Pipe()
    centroid_avg =Value('d', 0.0)
    quit = False
    all_processes = []
    img = xiapi.Image()
    print('Starting data acquisition...')
    cam.start_acquisition()
    if stackflag :
        print("Loading Stacks and ROI from :",stackflag)
        with open("stack.pkl", 'rb') as f:  
            load = pickle.load(f)
            stack = load[0]
            stackref = load[1]
            roimain = load[2]
            roiref = load[3]
            f.close()
    k= len(roimain)
    D = multiprocessing.Process(target=graphdisplayworker, args=[graph_q],daemon = True)
    R = multiprocessing.Process(target=record, args=[display_q,driftoutput_q,graph_q,k],daemon = True)


    print("SELECTED ROIs :",roimain,roiref)
    
    # print("Stack Size :",len(stack),len(stack[0]),len(stackref),len(stack[:][0][:]))
    cv2.destroyAllWindows()
    cv2.waitKey(2)
    print("Releasing Workers ...")
    for i in range(workers):
        p = multiprocessing.Process(target=worker, args=[input_q, output_q,stack],daemon = True)
        p.start()
        all_processes.append(p)
    for i in range(driftworkers):
        p = multiprocessing.Process(target=driftworker, args=[drift_q, driftoutput_q,stackref],daemon = True)
        p.start()
        all_processes.append(p)
    cv2.waitKey(2)
    R.start()
    D.start()
    
    fps = FPS().start()
    cv2.waitKey(2)
    frame_count=0
    print("Starting ...")
    try:
        while quit == False :
            timenow = time.time()
            cam.get_image(img)
            frame = img.get_image_data_numpy()
            frame = cv2.flip(frame, 0)  # flip the frame vertically
            frame = cv2.flip(frame, 1)
            for i in range(len(roimain)):
                roim=roimain[i]
                input_q.put([timenow,frame[int(roim[1]):int(roim[1]+roim[3]), int(roim[0]):int(roim[0]+roim[2])],i])
            drift_q.put([timenow,frame[int(roiref[1]):int(roiref[1]+roiref[3]), int(roiref[0]):int(roiref[0]+roiref[2])]])
            frame_count +=1
            if frame_count%50==0:
                k=1
                for j in roimain:
                    frame = cv2.rectangle(frame, (int(j[0]), int(j[1])), (int(j[0])+int(j[2]), int(j[1])+int(j[3])), (36,255,12), 2)
                    cv2.putText(frame, "Bead "+str(k), (int(j[0]), int(j[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    k+=1
                frame = cv2.rectangle(frame, (int(roiref[0]), int(roiref[1])), (int(roiref[0])+int(roiref[2]), int(roiref[1])+int(roiref[3])), (36,255,12), 2)
                cv2.putText(frame, "Ref Bead ", (int(roiref[0]), int(roiref[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                frame=cv2.resize(frame, (1125, 900),interpolation = cv2.INTER_NEAREST)
                cv2.imshow("Live",np.asarray(frame, dtype=np.uint8))
                cv2.waitKey(1)
            
            if output_q.empty():
                pass  # fill up queue
            else:
                for i in range(output_q.qsize()):
                    display_q.put((quit,output_q.get()))
                    fps.update()
                
    except KeyboardInterrupt:
        fps.stop()    
        quit = True
        time.sleep(4)
        cam.stop_acquisition()
        cam.close_device() 
        print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
        # print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
        os._exit(1)
