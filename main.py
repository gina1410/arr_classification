import streamlit as st
import numpy as np
from maindef import getModel, getPreProcessingSignal, getPrediction, plotPrediction

signals = np.array([], dtype='float')
fs = 360
pred_bool = False
model_path = 'CNN_model_17'

#---------------------------------------------
# Page layout
# Page expands to full width
st.set_page_config(
    page_title='Sistem Klasifikasi Aritmia',
    layout='wide'
)

# Page Intro
st.write("""
# Sistem Klasifikasi Aritmia

### Output tipe beat aritmia salah satu diantaranya yaitu:
- Normal beat (N)
- Left bundle branch block beat (LBBB)
- Right bundle branch block beat (RBBB)
- Premature ventricular contraction beat (PVC)
- Premature atrial contraction beat (PAC)

-------

""".strip())

#---------------------------------------------
# Sidebar - Collects user input features

with st.sidebar:
    pred_bool = False
    
    st.write('### Upload file sinyal EKG')
    uploaded_file = st.file_uploader('Upload file dengan format .csv atau .txt', type=['csv','txt'])
    
    if uploaded_file is not None:
        with st.sidebar.form(key ='Form1'):
            signals = np.loadtxt(uploaded_file, delimiter=',', skiprows=1)
            input_fs = st.number_input('Frekuensi sinyal EKG (Hz)')

            st.write('#### Terdapat ' + str(len(signals)) + ' sampel EKG')
            input_start_pred = st.number_input('Start point dari sampel EKG')
            input_end_pred = st.number_input('End point dari sampel EKG')

            submitted = st.form_submit_button('Mulai Klasifikasi')
            if submitted:
                fs = int(float(input_fs))
                start_pred = input_start_pred
                end_pred = input_end_pred
                pred_bool = True


#---------------------------------------------
# Main panel

model = getModel(model_path)

if pred_bool:
    signals = signals[int(float(start_pred)):int(float(end_pred))+1]
    
    with st.spinner(text='Sistem sedang memproses.....'):
        signals, beats, rpeaks = getPreProcessingSignal(signals, fs)
        pred_class = getPrediction(beats, model)

        fig = plotPrediction(signals, fs, rpeaks, pred_class)
        st.pyplot(fig)