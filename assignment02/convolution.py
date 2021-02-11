from scipy.io.wavfile import read
from scipy.signal import convolve
import numpy as np
import matplotlib.pyplot as plt
import time

# Answer to the question len(x) = 200, len(h) = 100. What is len(y)?
# Length of y would be: len(x) + len(h) - 1 = 299.
def myTimeConv(x,h):

    y = np.zeros(len(x) + len(h) - 1)

    padded_x = np.hstack((np.zeros(len(h)-1), x, np.zeros(len(h)-1)))
    reverse_h = np.flip(h)

    for i in range(len(y)):
        y[i] = np.dot(padded_x[i:i+len(h)], reverse_h)

    return y


def CompareConv(x, h):

    timer = np.zeros(2)

    current_time = time.time()
    my_conv = myTimeConv(x, h)
    timer[0] = time.time() - current_time

    current_time = time.time()
    scipy_conv = convolve(x, h)
    timer[1] = time.time() - current_time

    m = np.mean(my_conv - scipy_conv)
    mabs = np.mean(np.abs(my_conv - scipy_conv))
    stdev = np.std(my_conv - scipy_conv)

    return m, mabs, stdev, timer


def readwav(filename):
    fs, x = read(filename)
    if x.dtype == 'float32':
        audio = x
    else:
        # change range to [-1,1)
        if x.dtype == 'uint8':
            nbits = 8
        elif x.dtype == 'int16':
            nbits = 16
        elif x.dtype == 'int32':
            nbits = 32
        else:
            nbits = 1  # No conversion

        audio = x / float(2 ** (nbits - 1))

    # special case of unsigned format
    if x.dtype == 'uint8':
        audio = audio - 1.

    if audio.ndim > 1:
        audio = audio[:, 0]

    return audio


if __name__ == "__main__":

    x = np.ones(200)

    rampup = np.linspace(0,1, 26)
    rampdown = np.linspace(1,0,26)
    h = np.hstack((rampup[:-1],rampdown))

    y_time = myTimeConv(x, h)

    piano = readwav("piano.wav")
    impulse_response = readwav("impulse-response.wav")

    compConv = CompareConv(piano, impulse_response)

    fo = open("./results/02-convolutionStats.txt", "w+")

    fo.writelines("Mean: " + str(compConv[0]) + '\n')
    fo.writelines("Mean Absolute: " + str(compConv[1]) + '\n')
    fo.writelines("Standard Deviation: " + str(compConv[2]) + '\n')
    fo.writelines("Time of my convolution: " + str(compConv[3][0]) + '\n')
    fo.writelines("Time of Scipy convolution: " + str(compConv[3][1]) + '\n')

    time = np.linspace(0, len(y_time)-1, len(y_time))

    plt.plot(time, y_time)
    plt.xlabel("Time (In Samples)")
    plt.ylabel("Signal Amplitude")
    plt.title("Time Domain Convolution of two signals")
    plt.show()