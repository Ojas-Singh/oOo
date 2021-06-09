from imutils.video import FPS
from imutils.video import FileVideoStream
import imutils
import cv2
import numpy as np
import heapq
import time
import multiprocessing
from multiprocessing import Pool, Queue
import sys,os

quit = False
# RESIZE = 128
def worker(input_q, output_q):
    while True:
        frameinfo = input_q.get()
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
        time.sleep(.1)
        
        # output = imutils.resize(frameinfo[1], width=RESIZE, height=RESIZE)
        # output_q.put([frameinfo[0],output])
        output_q.put([frameinfo[0],frameinfo[1]])
        
def Stop(*args):
    quit = True

def displayworker(display_q,npsave,quit):
    cv2.namedWindow('Final')
    cv2.createButton('stop',Stop)
    while True:
        if quit:
            break
        frame = display_q.get()
        
        cv2.imshow('Final', frame)
        cv2.waitKey(1)
        





if __name__ == '__main__':
    qu_limit = 10
    workers = 12
    threadn = cv2.getNumberOfCPUs() 
    print("Threads : ", threadn)
    print("Workers Spawned : ", workers)
    input_q = Queue(qu_limit)  # fps is better if queue is higher but then more lags
    frame_count = 0
    stack=[]
    output_q = Queue()
    display_q = Queue()
    npsave = np.zeros([2,2])
    all_processes = []
    for i in range(workers):
        p = multiprocessing.Process(target=worker, args=[input_q, output_q],daemon = True)
        p.start()
        all_processes.append(p)
    D = multiprocessing.Process(target=displayworker, args=[display_q, npsave,quit],daemon = False)
    D.start()
    img = cv2.VideoCapture('sample.mp4')
    fps = FPS().start()
    ret, frame = img.read() 
    roi=cv2.selectROI(frame)
    cv2.destroyAllWindows()
    def getStack(*args):
        for i in range(200):
            ret, frame = img.read()
            stack.append(frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])])
            cv2.waitKey(1)
    cv2.namedWindow('GetStack')
    
    cv2.createButton('get',getStack)
    cv2.waitKey(0)
    
    while len(stack)< 201:
        ret, frame = img.read() 
        stack.append(frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])])
        cv2.waitKey(1)
    cv2.waitKey(2)
    fvs = FileVideoStream('sample.mp4').start()
    while quit == False and frame_count <500:
        
        frame = fvs.read()
        # if not fvs.more():
        #     print("nomore")
        #     break
        # ret, frame = img.read()  
        # if ret : 
            # roi_img = frame.copy()[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        input_q.put([time.time(),frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]])
        
    
        if output_q.empty():
            pass  # fill up queue
        else:
            frame_count += 1
            dummylist=[]
            for i in range(output_q.qsize()):
                dummylist.append(output_q.get())
            dummylist.sort()
            for i in dummylist:
                display_q.put(i[1])
            fps.update() 
            # print(frame_count)
        
                
        
    
    fps.stop()    
    quit =True
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
    D.terminate()
    for process in all_processes:
        process.terminate() 
    os._exit(1)
    # sys.exit()
    D.terminate()  
    for process in all_processes:
        process.terminate() 
    print('thread killed')
    

    
    
