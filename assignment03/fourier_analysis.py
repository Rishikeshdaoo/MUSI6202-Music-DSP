from scipy.io.wavfile import read
from scipy.signal import convolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import math


def generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):

    t = np.arange(0, length_secs, 1/sampling_rate_Hz)

    x = amplitude * np.sin(2 * np.pi * frequency_Hz * t + phase_radians)

    return t, x


def generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):

    t = np.arange(0, length_secs, 1/sampling_rate_Hz)
    x = np.zeros(len(t))

    for n in range(1,20,2):
        _, sine = generateSinusoidal(amplitude/n, sampling_rate_Hz, n * frequency_Hz, length_secs, phase_radians)
        x += sine

    return t, x


def computeSpectrum(x, sample_rate_Hz):

    N = len(x)

    if(N%2 == 0):
        N = N//2 + 1
    else:
        N = (N+1)//2 + 1

    nyquist = sample_rate_Hz/2.0
    dft = np.fft.fft(x)
    f = np.linspace(0.0, nyquist, N)
    XAbs = np.abs(dft[:N])
    XPhase = np.angle(dft[:N])
    XRe = dft.real
    XIm = dft.imag

    return f, XAbs, XPhase, XRe, XIm


def generateBlocks(x, sample_rate_Hz, block_size, hop_size):

    numBlocks = int(np.ceil(x.size / hop_size))
    X = np.zeros([numBlocks, block_size])
    t = (np.arange(numBlocks) * hop_size) / sample_rate_Hz
    x = np.concatenate((x, np.zeros(block_size)), axis=0)
    for n in range(numBlocks):
        i_start = n * hop_size
        i_stop = np.min([x.size - 1, i_start + block_size - 1])
        X[n][np.arange(0, block_size)] = x[np.arange(i_start, i_stop + 1)]

    X = X.T

    return t, X


def hann(L):
    return 0.5 - (0.5 * np.cos(2 * np.pi / L * np.arange(L))).reshape(1, -1)


def rect(L):
    return np.ones(L)


def mySpecgram(x, block_size, hop_size, sampling_rate_Hz, window_type):
    x = x.T


    afWindow = np.zeros(block_size)
    if(window_type == 'hann'):
        afWindow = np.hanning(block_size)
    elif(window_type == 'rect'):
        afWindow = rect(block_size)

    time_vector, xb = generateBlocks(x, sampling_rate_Hz, block_size, hop_size)

    xb = xb.T
    numBlocks = xb.shape[0]
    freq_vector = np.zeros([numBlocks, 1])

    magnitude_spectrogram = np.zeros([block_size//2+1, numBlocks])

    for n in range(0, numBlocks):
        # apply window
        win_sig = np.multiply(xb[n, :], afWindow)
        freq_vector, tmp, _, _, _ = computeSpectrum(win_sig, sampling_rate_Hz)
        magnitude_spectrogram[:,n] = tmp

    return freq_vector, time_vector, magnitude_spectrogram


if __name__ == "__main__":
    t, x_sine = generateSinusoidal(1.0, 44100, 400, 0.5, np.pi/2)
    t, x_square = generateSquare(1.0, 44100, 400, 0.5, 0)

    # plt.plot(t, x_sine)
    # plt.xlim(0, 0.005)
    # plt.xlabel("Time (In seconds)")
    # plt.ylabel("Amplitude")
    # plt.title("Sinusoid of frequency 400 Hz and phase pi/2")
    # plt.show()

    # plt.plot(t, x_square)
    # plt.xlim(0, 0.005)
    # plt.xlabel("Time (In seconds)")
    # plt.ylabel("Amplitude")
    # plt.title("Square wave of frequency 400 Hz")
    # plt.show()

    # f, XAbs, XPhase, _, _ = computeSpectrum(x_sine, 44100)

    # plt.figure(figsize=(10, 5))
    # plt.suptitle("Magnitude and phase spectrum for Sinusoid (400 Hz)")

    # plt.subplot(1,2,1)
    # plt.plot(f, XAbs)
    # plt.xlabel("Frequency (In Hz)")
    # plt.ylabel("Magnitude")
    # plt.subplot(1,2,2)
    # plt.plot(f, XPhase)
    # plt.xlabel("Frequency (In Hz)")
    # plt.ylabel("Phase (In Rad)")
    # plt.show()

    # f, XAbs, XPhase, _, _ = computeSpectrum(x_square, 44100)

    # plt.figure(figsize=(10, 5))
    # plt.suptitle("Magnitude and phase spectrum for Square wave (400 Hz)")

    # plt.subplot(1,2,1)
    # plt.plot(f, XAbs)
    # plt.xlabel("Frequency (In Hz)")
    # plt.ylabel("Magnitude")
    # plt.subplot(1,2,2)
    # plt.plot(f, XPhase)
    # plt.xlabel("Frequency (In Hz)")
    # plt.ylabel("Phase (In Rad)")
    # plt.show()

    freq_vector_rect, time_vector_rect, magnitude_spectrogram_rect = mySpecgram(x_square, 2046, 1024, 44100, 'rect')
    freq_vector_hann, time_vector_hann, magnitude_spectrogram_hann = mySpecgram(x_square, 2046, 1024, 44100, 'hann')

    # Taking log of the Magnitude spectrogram to plot the Spectrogram in dB scale
    logX_rect = 10 * np.log10(magnitude_spectrogram_rect)
    logX_hann = 10 * np.log10(magnitude_spectrogram_hann)

    fig = plt.figure(figsize=(20, 5))
    plt.subplot(1,2,1)
    plt.title("Spectrogram of Square wave with Rectangular Window")
    extent = time_vector_rect[0], time_vector_rect[-1], freq_vector_rect[-1], freq_vector_rect[0]
    plt.imshow(logX_rect, aspect='auto', extent=extent)
    plt.gca().invert_yaxis()
    fig.axes[0].set_xlabel('Time (s)')
    fig.axes[0].set_ylabel('Frequency (In log scale)')

    plt.subplot(1,2,2)
    plt.title("Spectrogram of Square wave with Von Hann Window")
    extent = time_vector_hann[0], time_vector_hann[-1], freq_vector_hann[-1], freq_vector_hann[0]
    plt.imshow(logX_hann, aspect='auto', extent=extent)
    plt.gca().invert_yaxis()
    fig.axes[1].set_xlabel('Time (s)')
    fig.axes[1].set_ylabel('Frequency (In log scale)')

    plt.show()

