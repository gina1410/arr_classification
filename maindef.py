import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from scipy.signal import medfilt, detrend, resample
from ecgdetectors.ecgdetectors import Detectors
from tensorflow import keras

#------------------------------------------------------

def denoise_signal(X, dwt_transform, dlevels, cutoff_low, cutoff_high):
    coeffs = pywt.wavedec(X, dwt_transform, level=dlevels)   # wavelet transform 'bior4.4'
    # scale 0 to cutoff_low 
    for ca in range(0, cutoff_low):
        coeffs[ca] = np.multiply(coeffs[ca], [0.0])
    # scale cutoff_high to end
    for ca in range(cutoff_high, len(coeffs)):
        coeffs[ca] = np.multiply(coeffs[ca], [0.0])
    Y = pywt.waverec(coeffs, dwt_transform) # inverse wavelet transform
    return Y  

#------------------------------------------------------

def get_median_filter_width(sampling_rate, duration):
    res = int( sampling_rate*duration )
    res += ((res%2) - 1) # needs to be an odd number
    return res

def filter_signal(X, FS):
    ms_flt_array = [0.2, 0.6]    #<-- length of baseline fitting filters (in seconds)
    mfa = np.zeros(len(ms_flt_array), dtype='int')
    for i in range(0, len(ms_flt_array)):
        mfa[i] = get_median_filter_width(FS, ms_flt_array[i])

    X0 = X  #read orignal signal
    for mi in range(0, len(mfa)):
        X0 = medfilt(X0, mfa[mi]) # apply median filter one by one on top of each other
    X0 = np.subtract(X, X0)  # finally subtract from orignal signal
    return X0

#------------------------------------------------------

def getPreProcessingSignal(signals, fs):   
    signals = detrend(signals)
    signals = denoise_signal(signals, 'bior4.4', 7 , 1 , 7)
    signals = filter_signal(signals, fs)
    signals = np.array(signals)

    # Detect R-Peaks
    detectors = Detectors(fs)
    rpeaks = detectors.christov_detector(signals)
    rpeaks = np.array(rpeaks)

    beats = np.empty_like(list(rpeaks), dtype=object)

    # Split into individual heartbeats
    for idx, idxval in enumerate(rpeaks):
        if (idx == 0):
            beats[idx] = signals[0:idxval+100]
        elif (idx == len(rpeaks) - 1):
            beats[idx] = signals[idxval-100:]
        else:
            beats[idx] = signals[rpeaks[idx-1]+50:rpeaks[idx+1]-50]
        
        # Resample
        if fs != 360:
            newsize = int((beats[idx].size * 360 / fs) + 0.5)
            beats[idx] = resample(beats[idx], newsize)
        
        beats[idx] = beats[idx][:440]
        
        # Normalize beat
        beats[idx] = (beats[idx] - beats[idx].min()) / beats[idx].ptp()

        # Pad with zeroes.
        zerocount = 440 - beats[idx].size
        beats[idx] = np.pad(beats[idx], (0, zerocount), 'constant', constant_values=(0.0, 0.0))

    return signals, beats, rpeaks

#------------------------------------------------------

def getModel(model_path):
    model = keras.models.load_model(model_path, compile=False)
    return model

#------------------------------------------------------

def getPrediction(beats, _model):
    df_pred = pd.DataFrame(list(beats[:]))

    pred = _model.predict(df_pred)
    pred_class = np.argmax(pred, axis=1)

    return pred_class

#------------------------------------------------------

def plotPrediction(signals, fs, rpeaks, pred_class):
    fig, ax = plt.subplots(figsize=(25,6))

    # Plot the heart beats. Time scale is number of readings
    # divided by sampling frequency.
    times = (np.arange(len(signals), dtype = 'float') + signals[0]) / fs
    ax.plot(times, signals)
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('Amplitude(mV)')

    # Extract annotations.
    annots = rpeaks[:len(pred_class)]

    # Plot the Annotations.
    annotimes = times[annots]
    ax.plot(annotimes, np.ones_like(annotimes) * signals.max() * 1.2, 'ro')

    # Annotation codes.
    for idx, annot in enumerate(annots):
        if pred_class[idx] == 0:
            annotype = 'N'
        elif pred_class[idx] == 1:
            annotype = 'LBBB'
        elif pred_class[idx] == 2:
            annotype = 'RBBB'
        elif pred_class[idx] == 3:
            annotype = 'PVC'
        elif pred_class[idx] == 4:
            annotype = 'PAC'

        ax.annotate(annotype, xy = (times[annot], signals.max() * 1.1))

    return fig