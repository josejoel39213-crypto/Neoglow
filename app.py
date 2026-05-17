# ============================================================
# NEOGLOW ONLINE DEPLOYMENT VERSION
# ============================================================

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time

from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NeoGlow AI Monitor",
    page_icon="🩺",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🩺 NeoGlow AI Neonatal Jaundice Monitoring")

st.write(
    """
AI-based neonatal bilirubin estimation and jaundice monitoring system.
"""
)

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 128

THRESHOLD = 0.6199821

CSV_FILE = "patient_monitoring.csv"

IMAGE_FOLDER = "patient_images"

os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_neoglow_model():

    model = load_model(
        "NeoGlow_Final_Model.keras"
    )

    return model

model = load_neoglow_model()

# ============================================================
# PATIENT INFORMATION
# ============================================================

st.sidebar.title("Patient Information")

patient_id = st.sidebar.text_input(
    "Patient ID"
)

patient_name = st.sidebar.text_input(
    "Patient Name"
)

if patient_id == "":

    patient_id = f"P{int(time.time())}"

    st.sidebar.info(
        f"Generated Patient ID: {patient_id}"
    )

# ============================================================
# WHITE BALANCE
# ============================================================

def white_balance(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    lower_yellow = np.array([15,50,50])

    upper_yellow = np.array([40,255,255])

    mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        x,y,w,h = cv2.boundingRect(largest)

        roi = img[y:y+h, x:x+w]

        if roi.size > 0:

            lab = cv2.cvtColor(
                roi,
                cv2.COLOR_BGR2LAB
            )

            avg_lab = np.mean(
                lab.reshape(-1,3),
                axis=0
            )

            full_lab = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2LAB
            ).astype(np.float32)

            target = np.array([200,128,128])

            diff = target - avg_lab

            full_lab[:,:,0] += diff[0]
            full_lab[:,:,1] += diff[1]
            full_lab[:,:,2] += diff[2]

            full_lab = np.clip(
                full_lab,
                0,
                255
            ).astype(np.uint8)

            img = cv2.cvtColor(
                full_lab,
                cv2.COLOR_LAB2BGR
            )

    return img

# ============================================================
# ROI EXTRACTION
# ============================================================

def extract_roi(img):

    h,w,_ = img.shape

    x1 = int(w*0.25)
    x2 = int(w*0.75)

    y1 = int(h*0.25)
    y2 = int(h*0.75)

    return img[y1:y2, x1:x2]

# ============================================================
# PREPROCESS
# ============================================================

def preprocess_frame(frame):

    img = frame.copy()

    img = white_balance(img)

    roi = extract_roi(img)

    roi = cv2.resize(
        roi,
        (IMG_SIZE,IMG_SIZE)
    )

    rgb = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB
    )

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB
    )

    rgb = rgb.astype(np.float32)/255.0
    hsv = hsv.astype(np.float32)/255.0
    lab = lab.astype(np.float32)/255.0

    rgb = rgb.reshape(1,-1,3)
    hsv = hsv.reshape(1,-1,3)
    lab = lab.reshape(1,-1,3)

    return rgb,hsv,lab

# ============================================================
# PREDICTION
# ============================================================

def predict_image(frame):

    rgb,hsv,lab = preprocess_frame(frame)

    pred_reg,pred_cls = model.predict([
        rgb,
        hsv,
        lab
    ],verbose=0)

    bilirubin = float(pred_reg[0][0])

    probability = float(pred_cls[0][0])

    prediction = (
        "JAUNDICE"
        if probability > THRESHOLD
        else "NORMAL"
    )

    return bilirubin,probability,prediction

# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    patient_id,
    patient_name,
    bilirubin,
    probability,
    prediction,
    image_path
):

    timestamp = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    row = {

        "Patient_ID": patient_id,

        "Patient_Name": patient_name,

        "Timestamp": timestamp,

        "Bilirubin": bilirubin,

        "Probability": probability,

        "Prediction": prediction,

        "Image_Path": image_path
    }

    df = pd.DataFrame([row])

    if not os.path.exists(CSV_FILE):

        df.to_csv(
            CSV_FILE,
            index=False
        )

    else:

        df.to_csv(
            CSV_FILE,
            mode='a',
            header=False,
            index=False
        )

# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown("---")

st.subheader("📁 Upload Neonatal Image")

uploaded_file = st.file_uploader(
    "Upload Neonatal Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        file_bytes,
        1
    )

    st.image(
        frame,
        channels='BGR',
        caption='Uploaded Image'
    )

    if st.button("Run AI Analysis"):

        bilirubin,probability,prediction = predict_image(frame)

        st.success("Analysis Complete")

        st.metric(
            "Predicted Bilirubin",
            f"{bilirubin:.2f} mg/dL"
        )

        st.metric(
            "Jaundice Probability",
            f"{probability:.4f}"
        )

        if prediction == "JAUNDICE":

            st.error(
                f"Prediction: {prediction}"
            )

        else:

            st.success(
                f"Prediction: {prediction}"
            )

        image_name = f"{patient_id}_{int(time.time())}.jpg"

        image_path = os.path.join(
            IMAGE_FOLDER,
            image_name
        )

        cv2.imwrite(
            image_path,
            frame
        )

        save_result(
            patient_id,
            patient_name,
            bilirubin,
            probability,
            prediction,
            image_path
        )

        st.success(
            "Patient data saved successfully"
        )

# ============================================================
# CAMERA INPUT
# ============================================================

st.markdown("---")

st.subheader("📷 Capture Using Camera")

camera_image = st.camera_input(
    "Take Neonatal Image"
)

if camera_image is not None:

    image = Image.open(camera_image)

    frame = np.array(image)

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    st.image(
        frame,
        channels='BGR',
        caption='Captured Image'
    )

    if st.button("Run AI on Captured Image"):

        bilirubin,probability,prediction = predict_image(frame)

        st.success("Analysis Complete")

        st.metric(
            "Predicted Bilirubin",
            f"{bilirubin:.2f} mg/dL"
        )

        st.metric(
            "Jaundice Probability",
            f"{probability:.4f}"
        )

        if prediction == "JAUNDICE":

            st.error(
                f"Prediction: {prediction}"
            )

        else:

            st.success(
                f"Prediction: {prediction}"
            )

        image_name = f"{patient_id}_{int(time.time())}.jpg"

        image_path = os.path.join(
            IMAGE_FOLDER,
            image_name
        )

        cv2.imwrite(
            image_path,
            frame
        )

        save_result(
            patient_id,
            patient_name,
            bilirubin,
            probability,
            prediction,
            image_path
        )

        st.success(
            "Patient data saved successfully"
        )

# ============================================================
# DASHBOARD
# ============================================================

st.markdown("---")

st.subheader("📊 Patient Monitoring Dashboard")

if os.path.exists(CSV_FILE):

    full_df = pd.read_csv(CSV_FILE)

    patient_df = full_df[
        full_df['Patient_ID'] == patient_id
    ]

    if len(patient_df) > 0:

        st.dataframe(patient_df)

        # ====================================================
        # IMPROVEMENT DETECTION
        # ====================================================

        if len(patient_df) >= 2:

            latest = patient_df.iloc[-1]['Bilirubin']

            previous = patient_df.iloc[-2]['Bilirubin']

            if latest < previous:

                st.success(
                    "📉 Improvement Detected"
                )

            elif latest > previous:

                st.error(
                    "📈 Bilirubin Increasing"
                )

            else:

                st.info(
                    "➖ Stable Bilirubin"
                )

        # ====================================================
        # GRAPH
        # ====================================================

        fig, ax = plt.subplots(figsize=(10,4))

        ax.plot(
            patient_df['Bilirubin'],
            marker='o'
        )

        ax.set_title(
            f'{patient_name} Bilirubin Trend'
        )

        ax.set_ylabel(
            'Bilirubin (mg/dL)'
        )

        ax.set_xlabel(
            'Reading Number'
        )

        st.pyplot(fig)

        # ====================================================
        # STORED IMAGES
        # ====================================================

        st.subheader("🖼 Stored Images")

        for i,row in patient_df.iterrows():

            col1, col2 = st.columns([5,1])

            with col1:

                if os.path.exists(row['Image_Path']):

                    st.image(
                        row['Image_Path'],
                        width=250,
                        caption=f"{row['Timestamp']} | Bilirubin: {row['Bilirubin']:.2f}"
                    )

            with col2:

                unique_key = f"{row['Patient_ID']}_{row['Timestamp']}_{i}"

                if st.button(
                    "❌",
                    key=unique_key
                ):

                    # ========================================
                    # DELETE IMAGE
                    # ========================================

                    if os.path.exists(row['Image_Path']):

                        os.remove(
                            row['Image_Path']
                        )

                    # ========================================
                    # DELETE CSV ROW
                    # ========================================

                    full_df = full_df.drop(i)

                    full_df.to_csv(
                        CSV_FILE,
                        index=False
                    )

                    st.success(
                        "Record Deleted"
                    )

                    st.rerun()

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.warning(
    "Research and screening use only. Clinical confirmation required."
)
