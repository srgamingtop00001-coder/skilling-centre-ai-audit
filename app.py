import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Skilling Centre AI Audit", layout="wide", page_icon="🏛️")

st.title("🏛️ Skilling Centre AI Monitoring & Discrepancy System")
st.write("Upload CCTV Frame / Image to perform AI Attendance Verification and Asset Audit.")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model 'best.pt': {e}. Please ensure 'best.pt' is in the repository root.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose a CCTV image...", type=["jpg", "jpeg", "png"])
    portal_attendance = st.number_input("Sanctioned Portal Attendance Record", min_value=0, value=25)
    run_btn = st.button("🔍 Run AI Audit Software", type="primary")

if run_btn and uploaded_file is not None and model_loaded:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    results = model(image)
    
    detected_persons = 0
    detected_chairs = 0
    other_assets = 0
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            if class_name.lower() == 'person':
                detected_persons += 1
            elif 'chair' in class_name.lower() or 'bench' in class_name.lower():
                detected_chairs += 1
            else:
                other_assets += 1
                
    annotated_frame = results[0].plot()
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    discrepancy = abs(portal_attendance - detected_persons)
    variance_pct = (discrepancy / portal_attendance * 100) if portal_attendance > 0 else 0
    
    with col2:
        st.image(annotated_frame_rgb, caption="AI Detection Output", use_column_width=True)
        st.subheader("📊 Audit Summary")
        st.write(f"**Registered Portal Attendance:** {portal_attendance}")
        st.write(f"**Physical Presence Detected (AI):** {detected_persons}")
        st.write(f"**Infrastructure Count:** Chairs/Benches={detected_chairs}, Tools={other_assets}")
        
        if variance_pct > 10:
            st.error(f"🚨 **STATUS: DISCREPANCY ALERT TRIGGERED!**\nVariance: {variance_pct:.1f}% mismatch found between physical presence and registered portal records.")
        else:
            st.success("✅ **STATUS: COMPLIANCE OK**\nAttendance matches portal record within acceptable tolerance.")
