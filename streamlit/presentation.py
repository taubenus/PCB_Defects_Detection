import h5py, os, cv2
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import classification_report
from collections import Counter

image_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir, 'data', 'Images_bb'))

st.title('Defect Detection on PCBs')
st.sidebar.title('Tabel of contents')

pages = ['Introduction', 'RES-UNET', 'Training of the Model', 'Results and Evaluation']
page = st.sidebar.radio('View page', pages)



if page == 'Introduction':
    st.write('Printed Circuit Boards (PCBs) are essential components in nearly all electronic devices. Ensuring their quality is critical, as defects can lead to device malfunctions or failures. Visual inspection, defect detection and recall are some of the most complex and time consuming tasks for PCB manufacturing companies. Over the years, Printed Circuit Boards have become much smaller and more densely packed with components making the scalability of visual inspection harder. Traditional inspection methods, often manual, are time-consuming and prone to human error.')

    options = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur_', 'spurious_copper']
    st.markdown('### Sample images with defects')
    choice = st.selectbox('Select Defect', options, index=None, placeholder='Choose a defect type', label_visibility='collapsed')
    if choice != None:
        img_pool_choice = [os.path.join(image_path, filename) for filename in os.listdir(image_path) if choice in filename]
        rnd_3 = np.random.choice(range(20), 3, replace=False)
        fig = plt.figure(figsize=(45, 15))
        for i,j in enumerate(rnd_3):
            img = cv2.imread(img_pool_choice[j], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(3, 1, i+1)
            plt.axis('off')
            plt.title(f"{choice} {i+1}")
            plt.imshow(img)
        st.pyplot(fig, use_container_width=False)


