from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def crossCorr(x, y):

    corr_length = len(x) + len(y) - 1
    corr = np.zeros(corr_length)
    padded_x = np.hstack((np.zeros((len(y)-1)), x, np.zeros(len(y) - 1)))

    for n in range(corr_length):
        corr[n] = np.dot(padded_x[n:n+len(y)], y)

    norm_corr = corr/np.max(corr)

    return norm_corr


def loadSoundFile(filename):
    _, audio = read(filename)

    if (audio.shape[1] > 1):
        return audio[:, 0]
    else:
        return audio


def findSnarePosition(snareFilename, drumloopFilename):

    snare = loadSoundFile(snareFilename)
    drums = loadSoundFile(drumloopFilename)

    corr = crossCorr(drums, snare)

    # Assumed a threshold of 0.8 would be a safe value after looking at the plot from Q.1
    corr_thresholded = np.where(corr >0.8, corr, 0)

    pos = argrelextrema(corr_thresholded, np.greater)

    # Subtracting the padding added in the beginning
    pos = np.ndarray.tolist(pos[0] - (len(snare) - 1))

    return pos


if __name__ == "__main__":
    loop = loadSoundFile("./drum_loop.wav")
    snare = loadSoundFile("./snare.wav")

    corr = crossCorr(loop, snare)
    lag = range(-len(snare)+1, len(loop))

    pos = findSnarePosition("./snare.wav", "./drum_loop.wav")

    plt.plot(lag, corr)
    plt.xlabel("Lag (In samples)")
    plt.ylabel("Normalized Correlation Coefficient")
    plt.title("Cross Correlation between Drum loop and Snare")
    plt.show()




