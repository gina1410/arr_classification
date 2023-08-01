import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import detrend, resample, butter, filtfilt
from tensorflow import keras

#------------------------------------------------------

# Butterworth LowPass Filter
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

#------------------------------------------------------

def getPreProcessingSignal(signals, fs):   
    signals = detrend(signals)
    signals = butter_lowpass_filter(signals, 50, fs, 2)
    signals = np.array(signals)

    # Detect R-Peaks
    signals_df, info = nk.ecg_peaks(signals, sampling_rate=360)
    rpeaks = info["ECG_R_Peaks"]

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

def plotPrediction(signals, rpeaks, pred_class):
    fig, ax = plt.subplots(figsize=(25,6))

    # Plot the heart beats.
    ax.plot(signals)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude(mV)')

    # Extract annotations.
    annots = rpeaks[:len(pred_class)]

    # Plot the Annotations.
    ax.plot(annots, np.ones_like(annots) * signals.max() * 1.2, 'ro')

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

        ax.annotate(annotype, xy = (annot, signals.max() * 1.1))

    return fig
