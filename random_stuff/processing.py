import numpy as np
from imutils.video import WebcamVideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
# import cupy as cp
import imutils
import cv2
import numba 
from numba import jit
import time
from threading import Thread
# from VideoShow import VideoShow

def ImageProcessing(frame,RESIZE):
    frame = imutils.resize(frame, width=RESIZE, height=RESIZE)
    #### Add Image Processing Task here.

    return frame

# @jit(nopython=True)
def Mathematics(frame,STACK,GPU):
    # if not GPU:
    frame = np.complex128(frame)
    f = np.fft.fft2(frame)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return 

print("[INFO] sampling THREADED frames from webcam...")
# vs = WebcamVideoStream(src=0).start()
fvs = FileVideoStream('sample.mp4').start()
fps = FPS().start()
RESIZE = 500
STACK=[1]
add_nulls = lambda number, zero_count : "{0:0{1}d}".format(number, zero_count)
prev_frame_time = 0
new_frame_time = 0


i=0
while True:
    print(i)
    frame = fvs.read()
    if not fvs.more():
        break
    frame = ImageProcessing(frame,RESIZE)
    # Mathematics(frame,STACK,False)
    # cv2.putText(frame, "Queue Size: {}".format(fps.fps()),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
    # show the frame and update the FPS counter

    # new_frame_time = time.time() 
    # fp = 1/(new_frame_time-prev_frame_time) 
    # prev_frame_time = new_frame_time 
    # fp = add_nulls(int(fp),3)
    # cv2.putText(frame, fp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    # video_shower.frame = frame
    fps.update()
    i+=1

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
# stream.release()
cv2.destroyAllWindows()