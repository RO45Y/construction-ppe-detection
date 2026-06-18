import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
from ultralytics import YOLO
import os
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="Construction Safety PPE Detection",
    page_icon="⛑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: bold;
    }
    .violation-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 5px 0;
    }
    .safe-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 5px 0;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Class names and colors for visualization
CLASS_NAMES = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 
    'NO-Safety Vest', 'Person', 'Safety Cone', 
    'Safety Vest', 'machinery', 'vehicle'
]

# Color mapping for classes (BGR format for OpenCV)
CLASS_COLORS = {
    'Hardhat': (0, 255, 0),          # Green
    'Mask': (0, 255, 0),             # Green
    'NO-Hardhat': (0, 0, 255),       # Red
    'NO-Mask': (0, 0, 255),          # Red
    'NO-Safety Vest': (0, 0, 255),   # Red
    'Person': (255, 255, 0),         # Cyan
    'Safety Cone': (255, 165, 0),    # Orange
    'Safety Vest': (0, 255, 0),      # Green
    'machinery': (128, 0, 128),      # Purple
    'vehicle': (255, 192, 203)       # Pink
}

@st.cache_resource
def load_model(model_path):
    """Load YOLOv8 model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection coordinates
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    # Intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def remove_duplicate_detections(detections, iou_threshold=0.5):
    """
    Remove duplicate detections using Non-Maximum Suppression (NMS).
    Keeps the highest confidence box when overlap is high.
    """
    if not detections:
        return detections
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    kept = []
    for det in detections:
        is_duplicate = False
        
        for kept_det in kept:
            # Only compare same class
            if det['class'] != kept_det['class']:
                continue
            
            # Calculate IoU
            iou = calculate_iou(det['bbox'], kept_det['bbox'])
            
            # If high overlap, it's a duplicate
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept.append(det)
    
    return kept

def analyze_detections(detections):
    """
    Simple analysis: count people, violations, and breakdown by type.
    """
    # Count all classes
    class_counts = Counter([d['class'] for d in detections])
    
    # Total people
    total_people = class_counts.get('Person', 0)
    
    # Count violations (NO-* classes)
    violations = {
        'NO-Hardhat': class_counts.get('NO-Hardhat', 0),
        'NO-Mask': class_counts.get('NO-Mask', 0),
        'NO-Safety Vest': class_counts.get('NO-Safety Vest', 0)
    }
    
    total_violations = sum(violations.values())
    
    # Count proper PPE worn
    ppe_worn = {
        'Hardhat': class_counts.get('Hardhat', 0),
        'Mask': class_counts.get('Mask', 0),
        'Safety Vest': class_counts.get('Safety Vest', 0)
    }
    
    # Estimate compliant people
    compliant_people = max(0, total_people - total_violations)
    
    return {
        'total_people': total_people,
        'compliant_people': compliant_people,
        'total_violations': total_violations,
        'violations_breakdown': violations,
        'ppe_worn': ppe_worn
    }

def display_dashboard(stats):
    """Display simplified dashboard with counts only"""
    st.markdown("---")
    st.subheader("📊 Safety Dashboard")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h2>{stats['total_people']}</h2>
            <p>👥 Total People</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h2>{stats['compliant_people']}</h2>
            <p>✅ Fully Compliant</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h2 style="color: #dc3545;">{stats['total_violations']}</h2>
            <p>⚠️ Total Violations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        rate = (stats['compliant_people'] / stats['total_people'] * 100) if stats['total_people'] > 0 else 0
        st.markdown(f"""
        <div class="metric-box">
            <h2>{rate:.0f}%</h2>
            <p>📈 Compliance Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Violations breakdown
    if stats['total_violations'] > 0:
        st.markdown("#### ⚠️ Missing PPE Breakdown")
        
        cols = st.columns(3)
        violation_data = [
            ('NO-Hardhat', '⛑️ Hardhat Missing', '#dc3545'),
            ('NO-Mask', '😷 Mask Missing', '#dc3545'),
            ('NO-Safety Vest', '🦺 Safety Vest Missing', '#dc3545')
        ]
        
        for i, (key, label, color) in enumerate(violation_data):
            count = stats['violations_breakdown'][key]
            with cols[i]:
                if count > 0:
                    st.markdown(f"""
                    <div class="violation-box">
                        <h3 style="color: {color}; margin: 0;">{count}</h3>
                        <p style="margin: 0; font-size: 0.9rem;">{label}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-box">
                        <h3 style="color: #28a745; margin: 0;">✓</h3>
                        <p style="margin: 0; font-size: 0.9rem;">{label.replace('Missing', 'OK')}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # PPE worn summary
    if stats['total_people'] > 0:
        st.markdown("#### ✅ PPE Properly Worn")
        cols = st.columns(3)
        ppe_data = [
            ('Hardhat', '⛑️ Hardhats', stats['ppe_worn']['Hardhat']),
            ('Mask', '😷 Masks', stats['ppe_worn']['Mask']),
            ('Safety Vest', '🦺 Safety Vests', stats['ppe_worn']['Safety Vest'])
        ]
        
        for i, (key, label, count) in enumerate(ppe_data):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-box">
                    <h3 style="color: #28a745; margin: 0;">{count}</h3>
                    <p style="margin: 0; font-size: 0.9rem;">{label}</p>
                </div>
                """, unsafe_allow_html=True)

def process_image(image, model, conf_threshold=0.25, iou_threshold=0.45):
    """Process single image and return annotated image and detections"""
    if isinstance(image, Image.Image):
        img_array = np.array(image)
        if img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_array = image
    
    # Run inference with IoU threshold
    results = model(img_array, conf=conf_threshold, iou=iou_threshold)
    
    # Extract detections
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = CLASS_NAMES[cls]
            
            detections.append({
                'class': class_name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            })
    
    # Remove duplicates
    detections = remove_duplicate_detections(detections, iou_threshold=0.5)
    
    # Draw boxes
    annotated_img = img_array.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class']
        conf = det['confidence']
        
        color = CLASS_COLORS.get(class_name, (0, 255, 0))
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + 10
        
        cv2.rectangle(annotated_img, (x1, label_y - label_size[1] - 5), 
                     (x1 + label_size[0], label_y + 5), color, -1)
        cv2.putText(annotated_img, label, (x1, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    return annotated_img_rgb, detections

def process_video_frame(frame, model, conf_threshold=0.25, iou_threshold=0.45):
    """Process a single video frame"""
    results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = CLASS_NAMES[cls]
            
            detections.append({
                'class': class_name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            })
    
    # Remove duplicates
    detections = remove_duplicate_detections(detections, iou_threshold=0.5)
    
    # Draw boxes
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class']
        conf = det['confidence']
        
        color = CLASS_COLORS.get(class_name, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + 10
        
        cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), 
                     (x1 + label_size[0], label_y + 5), color, -1)
        cv2.putText(frame, label, (x1, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame, detections

def main():
    st.markdown('<h1 class="main-header">⛑️ Construction Site Safety PPE Detection</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="best.pt",
        help="Path to your YOLOv8 best.pt model file"
    )
    
    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.25,
        step=0.05,
        help="Minimum confidence to keep a detection"
    )
    
    # IoU threshold for NMS
    iou_threshold = st.sidebar.slider(
        "NMS IoU Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.45,
        step=0.05,
        help="Lower = merge more overlapping boxes (reduce duplicates)"
    )
    
    st.sidebar.info(f"""
    **Quick Guide:**
    - **Duplicates?** Lower IoU to 0.30-0.40
    - **Missing objects?** Raise IoU to 0.55-0.65
    - **Default:** 0.45 works for most cases
    """)
    
    # Load model
    if os.path.exists(model_path):
        model = load_model(model_path)
        if model:
            st.sidebar.success("✅ Model loaded")
    else:
        st.sidebar.error("❌ Model not found")
        st.error(f"Please ensure `{model_path}` exists in the same directory or provide the correct path.")
        return
    
    # Input source selection
    st.sidebar.markdown("---")
    st.sidebar.title("📥 Input Source")
    source_option = st.sidebar.radio(
        "Select Source",
        ["Image Upload", "Video Upload", "Webcam"]
    )
    
    # Main content area
    if source_option == "Image Upload":
        st.subheader("📸 Image Detection")
        
        uploaded_file = st.file_uploader(
            "Upload an image", 
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("**Detection Result**")
                with st.spinner("Processing..."):
                    annotated_img, detections = process_image(
                        image, model, conf_threshold, iou_threshold
                    )
                    st.image(annotated_img, use_container_width=True)
            
            # Display dashboard
            stats = analyze_detections(detections)
            display_dashboard(stats)
    
    elif source_option == "Video Upload":
        st.subheader("🎥 Video Detection")
        
        uploaded_video = st.file_uploader(
            "Upload a video", 
            type=['mp4', 'avi', 'mov', 'mkv']
        )
        
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                st.error("Error opening video file")
                return
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            st.info(f"Video: {width}x{height} @ {fps}fps | Frames: {total_frames}")
            
            process_every = st.slider(
                "Process every N frames (1 = all frames)", 
                1, 10, 1
            )
            
            if 'stop_processing' not in st.session_state:
                st.session_state.stop_processing = False
            
            col1, col2 = st.columns([3, 1])
            with col1:
                start_button = st.button("🚀 Start Processing", type="primary")
            with col2:
                stop_button = st.button("⏹️ Stop", type="secondary")
            
            if stop_button:
                st.session_state.stop_processing = True
            
            if start_button:
                st.session_state.stop_processing = False
                progress_bar = st.progress(0)
                frame_placeholder = st.empty()
                dashboard_placeholder = st.empty()
                
                frame_count = 0
                latest_stats = None
                
                while cap.isOpened():
                    if st.session_state.stop_processing:
                        st.warning("Processing stopped by user")
                        break
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % process_every == 0:
                        processed_frame, detections = process_video_frame(
                            frame.copy(), model, conf_threshold, iou_threshold
                        )
                        
                        latest_stats = analyze_detections(detections)
                        
                        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        frame_placeholder.image(
                            processed_frame_rgb, 
                            channels="RGB", 
                            use_container_width=True
                        )
                        
                        # Update dashboard every 5 frames
                        if frame_count % (process_every * 5) == 0:
                            with dashboard_placeholder.container():
                                display_dashboard(latest_stats)
                        
                        progress = min((frame_count / total_frames), 1.0)
                        progress_bar.progress(progress)
                    
                    frame_count += 1
                
                cap.release()
                progress_bar.empty()
                
                # Final dashboard
                if latest_stats:
                    display_dashboard(latest_stats)
                
                if not st.session_state.stop_processing:
                    st.success("Video processing complete")
    
    elif source_option == "Webcam":
        st.subheader("🎥 Live Webcam Detection")
        
        webcam_id = st.number_input("Webcam ID", min_value=0, max_value=5, value=0)
        
        if 'webcam_running' not in st.session_state:
            st.session_state.webcam_running = False
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("▶️ Start Webcam", type="primary"):
                st.session_state.webcam_running = True
        with col2:
            if st.button("⏹️ Stop Webcam", type="secondary"):
                st.session_state.webcam_running = False
        
        if st.session_state.webcam_running:
            cap = cv2.VideoCapture(webcam_id)
            
            if not cap.isOpened():
                st.error("❌ Cannot open webcam. Please check your camera connection.")
                st.session_state.webcam_running = False
                return
            
            frame_placeholder = st.empty()
            dashboard_placeholder = st.empty()
            frame_count = 0
            start_time = time.time()
            
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame from webcam")
                    break
                
                processed_frame, detections = process_video_frame(
                    frame, model, conf_threshold, iou_threshold
                )
                
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                frame_placeholder.image(
                    processed_frame_rgb, 
                    channels="RGB", 
                    use_container_width=True
                )
                
                # Update dashboard every 10 frames
                if frame_count % 10 == 0:
                    stats = analyze_detections(detections)
                    with dashboard_placeholder.container():
                        display_dashboard(stats)
                        st.markdown(f"**Live FPS:** `{fps:.1f}`")
            
            cap.release()
            st.success("Webcam stopped")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit & YOLOv8</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()