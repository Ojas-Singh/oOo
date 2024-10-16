import cv2
import numpy as np
import math
#FKA Algo
I= np.zeros((256,256),dtype="uint8")
x0=129
y0=129
W=np.zeros((256,256),dtype="uint8")
for i in range(256):
    for j in range(256):
        W[i][j]= np.sqrt((i-x0)**2+(j-y0)**2)
print(W)
# a=12
# b=40
a=0
b=128
Rad=[]
for r in range(a,b):
    ff=[]
    for i in range(256):
        for j in range(256):
            if math.ceil(W[i][j]) == r:
                ff.append("I["+str(i)+"]"+"["+str(j)+"]")
    Rad.append(ff)

tt = open("FKA.py", "w")
tt.write("def FKA(I):\n")
tt.write("    R=[]\n")
a=5

for i in Rad:
    ot= False
    tt.write("    R.append((")
    d=0
    for j in i:
        if d<5000:
            if ot:
                tt.write("+"+str(j))
                d+=1
            else:
                tt.write(str(j))
                ot=True
                d+=1
    # tt.write(")/"+str(len(i))+".0)\n")
    tt.write(")/"+str(d)+".0)\n")
    # tt.write("/"+str(len(i))+"\n")
    a+=1
tt.write("    return R")
tt.close()