import streamlit as st
import pandas as pd
import os
import pytesseract
import cv2
import numpy as np
import re
from fpdf import FPDF
from datetime import datetime
from PIL import Image
import torch

# -----------------------
# 🔧 Configuration
# -----------------------
st.set_page_config(page_title="Smart Plate App", layout="wide")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

EXCEL_FILE = "number_plates.xlsx"
MODEL_PATH = "yolov5/runs/train/plate_detector/weights/best.pt"

# -----------------------
# 🧠 Load YOLO Model
# -----------------------
@st.cache_resource
def load_model():
    try:
        return torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

# -----------------------
# 🔍 Plate Recognition Logic
# -----------------------
def clean_text(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    return text if 6 <= len(text) <= 12 else None

def recognize_plate_yolo(image):
    results = model(image)
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        plate_img = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(plate_img, config='--oem 3 --psm 8')
        plate = clean_text(text)
        if plate:
            return plate, (x1, y1, x2 - x1, y2 - y1)
    return None, None

# -----------------------
# 🧾 Load or Create Excel
# -----------------------
if os.path.exists(EXCEL_FILE):
    df = pd.read_excel(EXCEL_FILE)
    if "Number Plate" not in df.columns or "Timestamp" not in df.columns:
        df = pd.DataFrame(columns=["Number Plate", "Timestamp"])
        df.to_excel(EXCEL_FILE, index=False)
else:
    df = pd.DataFrame(columns=["Number Plate", "Timestamp"])
    df.to_excel(EXCEL_FILE, index=False)

seen_plates = set(df["Number Plate"].astype(str).tolist())

# -----------------------
# 📄 PDF Export Function
# -----------------------
def export_to_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Number Plate Records", ln=True, align='C')
    for _, row in df.iterrows():
        pdf.cell(200, 10, txt=f"{row['Number Plate']} - {row['Timestamp']}", ln=True)
    pdf.output("Number_Plates_Report.pdf")

# -----------------------
# 🚀 Streamlit UI
# -----------------------
st.title("🚗 Number Plate Recognition System")
tab1, tab2, tab3 = st.tabs(["📸 Live Camera", "📊 Dashboard", "📤 Export"])

# 📸 Live Camera
with tab1:
    st.subheader("🔴 Real-time Detection")
    run = st.checkbox("Start Camera")
    frame_window = st.image([])

    if run and model:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                break

            plate_text, box = recognize_plate_yolo(frame)

            if plate_text and plate_text not in seen_plates:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df = pd.concat([df, pd.DataFrame([[plate_text, timestamp]], columns=["Number Plate", "Timestamp"])], ignore_index=True)
                df.to_excel(EXCEL_FILE, index=False)
                seen_plates.add(plate_text)
                st.toast(f"✅ Detected: {plate_text}")

            if box:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)
        cap.release()

# 📊 Dashboard
with tab2:
    st.subheader("📋 Plate Log Table")
    search_term = st.text_input("🔍 Search by Plate or Date")
    filtered_df = df[df.apply(lambda row: search_term.lower() in str(row["Number Plate"]).lower() or search_term in str(row["Timestamp"]), axis=1)] if search_term else df
    st.dataframe(filtered_df, use_container_width=True)

    if len(df):
        delete_index = st.number_input("🗑️ Delete row index", min_value=0, max_value=len(df) - 1, step=1)
        if st.button("Delete Selected Row"):
            df = df.drop(index=delete_index).reset_index(drop=True)
            df.to_excel(EXCEL_FILE, index=False)
            seen_plates = set(df["Number Plate"].tolist())
            st.success("✅ Row deleted successfully.")

# 📤 Export
with tab3:
    st.subheader("📂 Export Options")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Export to PDF"):
            export_to_pdf(df)
            with open("Number_Plates_Report.pdf", "rb") as file:
                st.download_button("⬇️ Download PDF", file, "Number_Plates_Report.pdf", mime="application/pdf")

    with col2:
        if st.button("📊 Export to CSV"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV", csv, "Number_Plates.csv", mime="text/csv")
