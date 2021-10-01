import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.pyplot import figure
with open('stackplot.pkl', 'rb') as f:
    # 118,132
    x0=96
    y0=90
    q1=0
    q2=-1
    a,b,c = pickle.load(f)
    # print(a[0].shape)
    l1,l2= a[0].shape
    fig, ax = plt.subplots(2,3)
    # print(type(c[0]))
    ax[0,0].imshow(b[q1], cmap=cm.Greys_r)
    x = [x0, l2]
    y = [y0, y0]
    ax[0,1].plot(x, y, color="red", linewidth=1)
    ax[0,1].imshow(a[q1], cmap=cm.Greys_r)
    x = np.arange(x0, x0+20)
    y = a[q1][y0][x0:x0+20]
    # print(a[0].shape)
    ax[0,2].set_yticks(np.arange(140, 240, 10)) 
    ax[0,2].set_xticks(np.arange(x0, x0+22, 2)) 
    ax[0,2].grid()
    ax[0,2].plot(x,y,'o-')
    x1=100
    x2=102
    x3=110
    ax[0,2].plot(x1, a[q1][y0][x1], marker='o', color="red")
    ax[0,2].plot(x2, a[q1][y0][x2], marker='o', color="red")
    ax[0,2].plot(x3, a[q1][y0][x3], marker='o', color="red")



    ax[1,0].imshow(b[q2], cmap=cm.Greys_r)
    x = [x0, l2]
    y = [y0, y0]
    ax[1,1].plot(x, y, color="red", linewidth=1)
    ax[1,1].imshow(a[q2], cmap=cm.Greys_r)
    x = np.arange(x0, x0+20)
    y = a[q2][y0][x0:x0+20]
    ax[1,2].set_yticks(np.arange(140, 240, 10)) 
    ax[1,2].set_xticks(np.arange(x0, x0+22, 2)) 
    ax[1,2].grid()
    ax[1,2].plot(x,y,'o-')
    # x1=98
    # x2=105
    # x3=115
    ax[1,2].plot(x1, a[q2][y0][x1], marker='o', color="red")
    ax[1,2].plot(x2, a[q2][y0][x2], marker='o', color="red")
    ax[1,2].plot(x3, a[q2][y0][x3], marker='o', color="red")
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('plot.png',dpi=100)
