import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import sys
import scipy.fft

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Check if command line argument is "-D"
        if sys.argv[1] == "-D":
            filename = "drift.txt"
        else:
            filename = str(sys.argv[1]) + ".txt"
        
        # Load data from file
        data = np.loadtxt("output/" + filename, dtype=str)

        x = data[:, 0]
        y = data[:, 1]
        ynew = []

        # Process data points
        for i in range(len(y)):
            if y[i] != 'None':
                d = [float(x[i]), float(y[i])]
                ynew.append(d)
            else:
                k = True
                p = 1
                while k:
                    if y[i + p] != 'None':
                        d = [float(x[i]), (float(ynew[-1][1]) + float(y[i + p])) / 2]
                        ynew.append(d)
                        k = False
                    else:
                        p += 1
        
        # Sort data points by x-coordinate
        ynew = sorted(ynew)
        x = [int(i[0] * 10000) for i in ynew]
        y = [i[1] for i in ynew]
        
        # Write processed data to file
        f = open("output/" + filename, "w")
        for i in ynew:
            f.write(str(i[0]) + " " + str(i[1]) + "\n")
        f.close()
        
        # Apply lowpass filter to y-values
        b, a = scipy.signal.butter(3, 0.005, 'lowpass')
        yn = scipy.signal.filtfilt(b, a, y, method="gust")

        # Calculate absolute difference between original and filtered y-values
        yfil = np.absolute(y - yn)
        f = [[], []]
        p = 0
        
        # Filter out data points with large absolute difference
        for i in range(len(y)):
            if np.absolute(y[i] - yn[i]) < 100:
                f[0].append(float(x[i] / 10000))
                f[1].append(y[i])
            else:
                p += 1
        
        print(len(x), p)
        
        # Plot original data points
        plt.scatter(x, y, c="black", s=0.2, alpha=0.8)
        plt.show()
        plt.clf()

        # Plot filtered data points
        plt.plot(f[0], f[1], 'k', alpha=0.9, linewidth=0.1)

        # Apply another lowpass filter to filtered y-values
        b, a = scipy.signal.butter(3, 0.01, 'lowpass')
        y2 = scipy.signal.filtfilt(b, a, f[1], method="gust")
        
        # Apply Savitzky-Golay filter to filtered y-values
        y1 = savgol_filter(f[1], 301, 4)
        
        # Plot filtered y-values
        plt.plot(f[0], y2, 'y--')
        plt.grid(axis='y')
        plt.show()
        plt.clf()
    else: 
        # If no command line argument is provided, open file dialog to select file
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename()
        data = np.loadtxt(file_path, dtype=str)

        x = data[:, 0]
        y = data[:, 1]
        ynew = []

        # Process data points
        for i in range(len(y)):
            if y[i] != 'None':
                d = [float(x[i]), float(y[i])]
                ynew.append(d)
            else:
                k = True
                p = 1
                while k:
                    if y[i + p] != 'None':
                        d = [float(x[i]), (float(ynew[-1][1]) + float(y[i + p])) / 2]
                        ynew.append(d)
                        k = False
                    else:
                        p += 1
        
        # Sort data points by x-coordinate
        ynew = sorted(ynew)
        x = [int(i[0] * 10000) for i in ynew]
        y = [i[1] for i in ynew]
        
        # Write processed data to file
        f = open(file_path, "w")
        for i in ynew:
            f.write(str(i[0]) + " " + str(i[1]) + "\n")
        f.close()
        
        # Apply lowpass filter to y-values
        b, a = scipy.signal.butter(3, 0.005, 'lowpass')
        yn = scipy.signal.filtfilt(b, a, y, method="gust")

        # Calculate absolute difference between original and filtered y-values
        yfil = np.absolute(y - yn)
        f = [[], []]
        p = 0
        
        # Filter out data points with large absolute difference
        for i in range(len(y)):
            if np.absolute(y[i] - yn[i]) < 100:
                f[0].append(float(x[i] / 10000))
                f[1].append(y[i])
            else:
                p += 1
        
        print(len(x), p)
        
        # Plot original data points
        plt.scatter(x, y, c="black", s=0.2, alpha=0.8)
        plt.show()
        plt.clf()

        # Plot filtered data points
        plt.plot(f[0], f[1], 'k', alpha=0.9, linewidth=0.1)

        # Apply another lowpass filter to filtered y-values
        b, a = scipy.signal.butter(3, 0.005, 'lowpass')
        y2 = scipy.signal.filtfilt(b, a, f[1], method="gust")
        
        # Apply Savitzky-Golay filter to filtered y-values
        y1 = savgol_filter(f[1], 301, 4)
        
        # Plot filtered y-values
        plt.plot(f[0], y2, 'y--')
        plt.grid(axis='y')
        plt.show()
        plt.clf()