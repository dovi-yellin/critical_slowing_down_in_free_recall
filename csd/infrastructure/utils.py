from scipy import signal

# utils for filtering
def butter_filter(cutoff, fs, order=5, btype='low'):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False, output='ba')
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_filter(cutoff, fs, order=order, btype='low')
    y = signal.lfilter(b, a, data) # lfilter
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_filter(cutoff, fs, order=order, btype='high')
    y = signal.lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.sosfilt(sos, data)
        return y