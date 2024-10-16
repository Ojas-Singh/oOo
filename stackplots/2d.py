import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.pyplot import figure
import math
import FKA
from matplotlib.cm import coolwarm,plasma


mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.rcParams["figure.figsize"] = (10,6)

#plt.yaxis.set_major_locator(mpl.ticker.MultipleLocator(200))
with open('stack.pkl', 'rb') as f:
    a = pickle.load(f)
        #5= bead
    #6 = kaizer
    #7 = BeadX Kaiser
    #8 = Fft_magnitude
    image=a[5][-1]


    fig = plt.figure()

    ax = plt.axes()

    ax.imshow(image)
    # ax.contour3D(X, Y, z,cmap ='binary',alpha=0.8)
    # ax.scatter(X, Y, z, c=z, cmap='viridis', linewidth=0.5);
    # ax.plot_wireframe(X, Y, z, cmap='plasma',linewidth=0.2)
    #ax.plot_surface(image[], Y, z, rstride=1, cstride=1,cmap='magma', edgecolor='none',alpha=0.4)
    #ax.view_init(35, 35)
    fig.set_size_inches(8, 8)
    fig.savefig('plot2d.png',dpi=450)