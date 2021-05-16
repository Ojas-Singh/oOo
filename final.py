from imutils.video import FPS
import imutils
import cv2
import numpy as np
import heapq
import time
import multiprocessing
from multiprocessing import Pool, Queue


RESIZE = 100
def worker(input_q, output_q):
    while True:
        frameinfo = input_q.get()
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
        # time.sleep(.05)
        
        # output = imutils.resize(frameinfo[1], width=RESIZE, height=RESIZE)
        # output_q.put([frameinfo[0],output])
        output_q.put([frameinfo[0],frameinfo[1]])


def displayworker(display_q,npsave):
    while True:
        frame = display_q.get()
        cv2.imshow('Video', frame)
        cv2.waitKey(1)




if __name__ == '__main__':
    qu_limit = 50
    threadn = cv2.getNumberOfCPUs() 
    print("Threads : ", threadn)
    input_q = Queue(qu_limit)  # fps is better if queue is higher but then more lags
    # input_q= heapq.heapify(input_q)
    output_q = Queue()
    display_q = Queue()
    npsave = np.zeros([2,2])
    for i in range(50):
        p = multiprocessing.Process(target=worker, args=[input_q, output_q])
        p.start()
    D = multiprocessing.Process(target=displayworker, args=[display_q, npsave])
    D.start()

    img = cv2.VideoCapture('sample.mp4')
    fps = FPS().start()
    frame_count = 0
    while True and frame_count< 1000:
        
        ret, frame = img.read()             
        # if frame_count % qu_limit == 0:            
        #     input_q.put(frame)        
        input_q.put([time.time(),frame])
        
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
            # data = output_q.get()[1] 
            # display_q.put(data)
            fps.update()               
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    fps.stop()    
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
   

    cv2.destroyAllWindows()
    