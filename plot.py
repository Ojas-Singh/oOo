import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.pyplot import figure
# gradients based on http://bsou.io/posts/color-gradients-with-python

def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
    ''' [255,255,255] -> "#FFFFFF" '''
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#"+"".join(["0{0:x}".format(v) if v < 16 else
                        "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
    ''' Takes in a list of RGB sub-lists and returns dictionary of
      colors in RGB and hex form for use in a graphing function
      defined later on '''
    return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
            "r":[RGB[0] for RGB in gradient],
            "g":[RGB[1] for RGB in gradient],
            "b":[RGB[2] for RGB in gradient]}
    # return [(rgb[0],rgb[1],rgb[2]) for rgb in gradient]

def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    ''' returns a gradient list of (n) colors between
      two hex colors. start_hex and finish_hex
      should be the full six-digit color string,
      inlcuding the number sign ("#FFFFFF") '''
    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
            int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
            for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)

    return color_dict(RGB_list)


def polylinear_gradient(colors, n):
    ''' returns a list of colors forming linear gradients between
        all sequential pairs of colors. "n" specifies the total
        number of desired output colors '''
    # The number of colors per individual linear gradient
    n_out = int(float(n) / (len(colors) - 1))
    # returns dictionary defined by color_dict()
    gradient_dict = linear_gradient(colors[0], colors[1], n_out)

    if len(colors) > 1:
        for col in range(1, len(colors) - 1):
            next = linear_gradient(colors[col], colors[col+1], n_out)
            for k in ("hex", "r", "g", "b"):
                # Exclude first point to avoid duplicates
                gradient_dict[k] += next[k][1:]

    return gradient_dict

fact_cache = {}
def fact(n):
    ''' Memoized factorial function '''
    try:
        return fact_cache[n]
    except(KeyError):
        if n == 1 or n == 0:
            result = 1
        else:
            result = n*fact(n-1)
        fact_cache[n] = result
        return result


def bernstein(t,n,i):
    ''' Bernstein coefficient '''
    binom = fact(n)/float(fact(i)*fact(n - i))
    return binom*((1-t)**(n-i))*(t**i)


def bezier_gradient(colors, n_out=100):
    ''' Returns a "bezier gradient" dictionary
        using a given list of colors as control
        points. Dictionary also contains control
        colors/points. '''
    # RGB vectors for each color, use as control points
    RGB_list = [hex_to_RGB(color) for color in colors]
    n = len(RGB_list) - 1

    def bezier_interp(t):
        ''' Define an interpolation function
            for this specific curve'''
        # List of all summands
        summands = [
            map(lambda x: int(bernstein(t,n,i)*x), c)
            for i, c in enumerate(RGB_list)
        ]
        # Output color
        out = [0,0,0]
        # Add components of each summand together
        for vector in summands:
            for c in range(3):
                out[c] += vector[c]

        return out

    gradient = [
        bezier_interp(float(t)/(n_out-1))
        for t in range(n_out)
    ]
    # Return all points requested for gradient
    return color_dict(gradient)


# color_dict1 = linear_gradient("#FFC300","#581845")


# color_dict1 = bezier_gradient  (("#07467A","#6D027C","#A0B800","#BD6D00","#07467A"),400)
c1='red' #blue
c2='yellow' #green
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))

    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
def COLOR(i):
    r=float((400-i*0.5)/400)
    g=float((1+i)/400)
    b=float((1+i*0.5)/400)
    return (r,g,b)
    # return (0.5,0.5,0.5)
with open('stackplot.pkl', 'rb') as f:

    # 118,132
    x0=96
    y0=90
    q1=0
    q2=-1
    a,b,c = pickle.load(f)
    print(len(a))
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
    plt.savefig('plot.png',dpi=300)

    plt.clf()
    color_dict1=polylinear_gradient(("#0000FF","#FF0000","#FFFF00"), 450)
    # color_dict1=bezier_gradient(("#07467A","#6D027C","#A0B800","#BD6D00","#07467A"),450)
    x = np.arange(x0, x0+50)
    for l in range(0,len(a),12):

        i=l
        # print(i)
        # plt.plot(x,a[i][y0][x0:x0+10],c=colorFader(c1,c2,i/400),alpha=0.5)
        plt.plot(x,a[i][y0][x0:x0+50],c=color_dict1.get('hex')[i],alpha=0.4)
        # plt.fill_between(x, y1=a[i][y0][x0:x0+10], y2=170, alpha=0.4, color=color_dict1.get('hex')[i], linewidth=1)
    plt.plot()
    plt.savefig('plotfull.png',dpi=300)
    plt.clf()
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    x = np.arange(0, l2)
    y = np.arange(0, l1)
    z = np.array(a[q2][:][:])
    # x = np.arange(50, 150)
    # y = np.arange(50, 150)
    X, Y = np.meshgrid(x,y)
    
    color_dict1=polylinear_gradient(("#0000FF","#FF0000","#FFFF00"), 450)
    
    # ax.contour3D(X, Y, z,cmap ='binary',alpha=0.8)
    # ax.scatter(X, Y, z, c=z, cmap='viridis', linewidth=0.5);
    # ax.plot_wireframe(X, Y, z, cmap='plasma',linewidth=0.2)
    ax.plot_surface(X, Y, z, rstride=1, cstride=1,
                cmap='magma', edgecolor='none',alpha=0.4)
    ax.view_init(35, 35)
    fig.savefig('plot3d.png',dpi=300)
