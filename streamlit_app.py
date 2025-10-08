import os
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import tempfile
import datetime
import csv
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from collections import Counter, deque
import requests
from typing import List, Dict, Tuple, Optional

# Configure page first
st.set_page_config(
    page_title="♻️ AI Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2ecc71;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #27ae60;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    .recycling-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    .stat-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .detection-box {
        border: 3px solid #3498db;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Helper Functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def speak_text(text: str, lang: str = 'en') -> str:
    """Convert text to speech with caching."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)
        return tmp_file.name
    except Exception as e:
        st.warning(f"Text-to-speech failed: {e}")
        return ""

def save_feedback(predicted_class: str, actual_class: str, confidence: float, 
                  image_info: str, feedback_notes: str = "", user_rating: int = 0):
    """Save user feedback to CSV with enhanced metadata."""
    feedback_file = "feedback.csv"
    file_exists = os.path.isfile(feedback_file)
    
    try:
        with open(feedback_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'predicted_class', 'actual_class', 'confidence',
                    'image_info', 'feedback_notes', 'user_rating', 'session_id'
                ])
            
            writer.writerow([
                datetime.datetime.now().isoformat(),
                predicted_class,
                actual_class,
                confidence,
                image_info,
                feedback_notes,
                user_rating,
                st.session_state.get('session_id', 'unknown')
            ])
    except Exception as e:
        st.error(f"Failed to save feedback: {e}")

def create_confidence_plot(predictions: List[Tuple[str, float]]) -> go.Figure:
    """Create an interactive confidence plot using Plotly."""
    classes, confidences = zip(*predictions)
    
    fig = go.Figure(data=[
        go.Bar(
            x=confidences,
            y=classes,
            orientation='h',
            marker_color=['#2ecc71' if conf == max(confidences) else '#3498db' for conf in confidences],
            text=[f'{conf*100:.1f}%' for conf in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence Scores",
        xaxis_title="Confidence",
        yaxis_title="Waste Categories",
        height=400,
        showlegend=False
    )
    
    return fig

def create_detection_visualization(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Create enhanced visualization with bounding boxes and labels."""
    img_annotated = image.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']
        
        # Get color based on class
        color = waste_detector.get_color(detection['class_idx'])
        
        # Draw bounding box
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img_annotated, (x1, y1 - label_size[1] - 15), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(img_annotated, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw confidence bar
        bar_width = x2 - x1
        confidence_width = int(bar_width * confidence)
        cv2.rectangle(img_annotated, (x1, y2), (x1 + confidence_width, y2 + 8), color, -1)
    
    return img_annotated

# Enhanced Waste Detector Class
class WasteDetector:
    def __init__(self, model, class_names: List[str], confidence_threshold: float = 0.5):
        self.model = model
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.detection_history = deque(maxlen=100)  # Track recent detections
    
    def detect_objects(self, image: Image.Image) -> List[Dict]:
        """Enhanced object detection with multiple strategies."""
        img = np.array(image)
        original_h, original_w = img.shape[:2]
        
        # Adaptive window sizing
        min_dim = min(original_h, original_w)
        window_size = min(224, min_dim)
        stride = max(56, window_size // 2)  # Adaptive stride
        
        detections = []
        
        # Multi-scale detection
        scales = [1.0, 0.75, 0.5] if min_dim > 300 else [1.0, 0.5]
        
        for scale in scales:
            scaled_w = int(original_w * scale)
            scaled_h = int(original_h * scale)
            
            if scaled_w < window_size or scaled_h < window_size:
                continue
                
            scaled_img = cv2.resize(img, (scaled_w, scaled_h))
            
            for y in range(0, scaled_h - window_size + 1, stride):
                for x in range(0, scaled_w - window_size + 1, stride):
                    window = scaled_img[y:y+window_size, x:x+window_size]
                    
                    # Predict
                    window_pil = Image.fromarray(window)
                    window_processed = self.preprocess_image(window_pil)
                    preds = self.model.predict(window_processed, verbose=0)
                    confidence = float(np.max(preds))
                    class_idx = int(np.argmax(preds, axis=1)[0])
                    
                    if confidence > self.confidence_threshold:
                        # Scale coordinates back to original
                        orig_x = int(x / scale)
                        orig_y = int(y / scale)
                        orig_size = int(window_size / scale)
                        
                        detections.append({
                            'bbox': (orig_x, orig_y, orig_x + orig_size, orig_y + orig_size),
                            'class': self.class_names[class_idx],
                            'confidence': confidence,
                            'class_idx': class_idx,
                            'scale': scale
                        })
        
        # Apply non-maximum suppression
        detections = self.non_max_suppression(detections)
        
        # Update detection history
        for detection in detections:
            self.detection_history.append(detection)
        
        return detections
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Enhanced preprocessing with data augmentation for robustness."""
        img_resized = image.resize((224, 224))
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    def non_max_suppression(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Enhanced NMS with class-aware suppression."""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections of same class
            detections = [
                det for det in detections 
                if not (det['class_idx'] == current['class_idx'] and 
                       self.iou(current['bbox'], det['bbox']) > iou_threshold)
            ]
        
        return keep
    
    def iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def get_color(self, class_idx: int) -> Tuple[int, int, int]:
        """Get distinct color for each class."""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (255, 165, 0), (255, 192, 203), (138, 43, 226)
        ]
        return colors[class_idx % len(colors)]
    
    def get_detection_stats(self) -> Dict:
        """Get statistics from detection history."""
        if not self.detection_history:
            return {}
        
        classes = [det['class'] for det in self.detection_history]
        confidences = [det['confidence'] for det in self.detection_history]
        
        return {
            'total_detections': len(self.detection_history),
            'most_common_class': Counter(classes).most_common(1)[0] if classes else ('None', 0),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'class_distribution': dict(Counter(classes))
        }

# Enhanced Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_update_time = time.time()
        self.update_interval = 1.5  # Faster updates
        self.last_detections = []
        self.frame_count = 0
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Process every few frames to balance performance and responsiveness
        current_time = time.time()
        if current_time - self.last_update_time > self.update_interval:
            self.last_update_time = current_time
            
            try:
                # Convert to PIL for processing
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                # Detect objects
                detections = waste_detector.detect_objects(pil_img)
                self.last_detections = detections
                
            except Exception as e:
                st.warning(f"Detection error: {e}")
        
        # Always draw the latest detections
        if self.last_detections:
            img = create_detection_visualization(img, self.last_detections)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Enhanced Recycling Information
recycling_info = {
    "Electronic waste": {
        "status": "Special Handling Required",
        "notes": "E-waste contains hazardous materials and valuable components. Take to certified e-waste recycling centers or manufacturer take-back programs.",
        "icon": "🔌",
        "color": "#e74c3c",
        "disposal_guide": "Do not dispose in regular trash. Find e-waste centers via local municipality or retailer programs."
    },
    "cardboard waste": {
        "status": "Recyclable",
        "notes": "Flatten boxes and keep clean/dry. Remove tape and labels. Greasy cardboard goes in compost.",
        "icon": "📦",
        "color": "#3498db",
        "disposal_guide": "Place in recycling bin. Ensure dry and clean condition."
    },
    "clothe waste": {
        "status": "Reusable/Recyclable",
        "notes": "Donate wearable items. Unwearable textiles go to textile recycling programs.",
        "icon": "👕",
        "color": "#9b59b6",
        "disposal_guide": "Donation centers for good condition, textile recycling for damaged items."
    },
    "glass waste": {
        "status": "Recyclable",
        "notes": "Rinse bottles/jars, remove lids. No window glass, mirrors, or ceramics.",
        "icon": "🍶",
        "color": "#1abc9c",
        "disposal_guide": "Separate by color if required. Place in recycling bin."
    },
    "metal waste": {
        "status": "Highly Recyclable",
        "notes": "Empty and rinse cans. Large metal to scrap yards. Excellent recycling value.",
        "icon": "🥫",
        "color": "#f39c12",
        "disposal_guide": "Curbside recycling for cans, scrap yards for large pieces."
    },
    "organic waste": {
        "status": "Compostable",
        "notes": "Food scraps and yard waste. Creates valuable compost. Reduces landfill waste.",
        "icon": "🍎",
        "color": "#27ae60",
        "disposal_guide": "Compost bin or green waste collection. Never in recycling."
    },
    "paper waste": {
        "status": "Recyclable",
        "notes": "Clean, dry paper only. Shredded paper may need special handling.",
        "icon": "📄",
        "color": "#34495e",
        "disposal_guide": "Recycling bin. Keep dry and remove contaminants."
    },
    "plastic waste": {
        "status": "Conditionally Recyclable",
        "notes": "Check resin codes (#1-7). Bottles/jugs typically accepted. Bags need special drop-off.",
        "icon": "🧴",
        "color": "#e67e22",
        "disposal_guide": "Check local guidelines. Typically #1, #2, #5 accepted."
    },
    "shoes waste": {
        "status": "Special Recycling",
        "notes": "Donate wearable pairs. Specialized recycling for unwearable shoes.",
        "icon": "👟",
        "color": "#8e44ad",
        "disposal_guide": "Shoe donation bins or manufacturer take-back programs."
    },
    "trash": {
        "status": "Landfill",
        "notes": "Non-recyclable, contaminated, or mixed waste. Try to reduce and separate first.",
        "icon": "🗑️",
        "color": "#7f8c8d",
        "disposal_guide": "Regular trash bin. Consider waste reduction strategies."
    }
}

# Initialize session state with enhanced tracking
def initialize_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(datetime.datetime.now().timestamp())
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'realtime_detections' not in st.session_state:
        st.session_state.realtime_detections = []
    if 'last_detection_time' not in st.session_state:
        st.session_state.last_detection_time = 0
    if 'user_feedback' not in st.session_state:
        st.session_state.user_feedback = []
    if 'app_usage_stats' not in st.session_state:
        st.session_state.app_usage_stats = {
            'total_predictions': 0,
            'total_detections': 0,
            'favorite_category': None,
            'start_time': datetime.datetime.now()
        }

# Load model and class names
@st.cache_resource
def load_cached_model():
    try:
        model = load_model(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

@st.cache_data
def get_class_names():
    try:
        datagen = ImageDataGenerator(rescale=1./255)
        tmp_gen = datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=1,
            class_mode='categorical',
            shuffle=False
        )
        class_indices = tmp_gen.class_indices
        idx_to_class = {v: k for k, v in class_indices.items()}
        return [idx_to_class[i] for i in range(len(idx_to_class))]
    except Exception as e:
        st.error(f"❌ Failed to load class names: {e}")
        return [f"Class_{i}" for i in range(10)]

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "final_waste_model_resnet.keras")
train_dir = os.path.join(BASE_DIR, "data", "split", "train")
examples_dir = os.path.join(BASE_DIR, "recycling_examples")

# Initialize app
initialize_session_state()

# Load model and setup detector
model = load_cached_model()
class_names = get_class_names()

if model is not None:
    waste_detector = WasteDetector(model, class_names, confidence_threshold=0.6)
else:
    st.error("🚫 Critical Error: Model failed to load. Please check the model file.")
    st.stop()

# ============================
# HELPER FUNCTION DEFINITIONS
# ============================

def collect_user_feedback(predicted_class, confidence):
    """Collect user feedback for predictions."""
    st.markdown("---")
    st.markdown("#### 💬 Help Improve Our AI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        actual_class = st.selectbox(
            "What type of waste is this really?",
            options=[""] + class_names,
            key=f"actual_class_{predicted_class}_{confidence}"
        )
    
    with col2:
        user_rating = st.slider(
            "How accurate was this prediction?",
            min_value=1,
            max_value=5,
            value=3,
            key=f"rating_{predicted_class}_{confidence}"
        )
    
    feedback_notes = st.text_area(
        "Additional notes (optional):",
        placeholder="Any additional feedback or comments...",
        key=f"notes_{predicted_class}_{confidence}"
    )
    
    if st.button("Submit Feedback", key=f"submit_{predicted_class}_{confidence}"):
        if actual_class:
            save_feedback(
                predicted_class=predicted_class,
                actual_class=actual_class,
                confidence=confidence,
                image_info="User feedback",
                feedback_notes=feedback_notes,
                user_rating=user_rating
            )
            st.success("✅ Thank you for your feedback! It helps improve our AI.")
            
            # Store in session state
            st.session_state.user_feedback.append({
                'timestamp': datetime.datetime.now(),
                'predicted_class': predicted_class,
                'actual_class': actual_class,
                'user_rating': user_rating
            })
        else:
            st.warning("⚠️ Please select the actual waste type.")

def display_single_prediction_results(predictions):
    """Display results for single image classification."""
    best_class, best_confidence = predictions[0]
    info = recycling_info.get(best_class, {})
    
    st.markdown("### 🤖 Recycling Assistant")
    
    # Main prediction card
    st.markdown(f"""
    <div class="prediction-card">
        <h3>{info.get('icon', '❓')} {best_class}</h3>
        <p><strong>Confidence:</strong> {best_confidence:.1%}</p>
        <p><strong>Status:</strong> {info.get('status', 'Unknown')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recycling guidance card
    st.markdown(f"""
    <div class="recycling-card">
        <h4>♻️ Recycling Guidance</h4>
        <p>{info.get('notes', 'No specific guidance available.')}</p>
        <p><strong>Disposal:</strong> {info.get('disposal_guide', 'Check local regulations.')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence visualization
    if st.session_state.get('show_confidence', True):
        st.plotly_chart(create_confidence_plot(predictions), use_container_width=True)
    
    # Audio feedback
    if st.session_state.get('enable_audio', True):
        speech_text = f"This appears to be {best_class}. It is {info.get('status', '')}. {info.get('notes', '')}"
        speech_file = speak_text(speech_text)
        if speech_file:
            st.audio(speech_file)

def perform_single_classification(img, display_col):
    """Perform single image classification when no objects detected."""
    with display_col:
        st.warning("⚠️ No specific objects detected. Performing general classification...")
        
        # Preprocess and predict
        img_processed = waste_detector.preprocess_image(img)
        preds = model.predict(img_processed, verbose=0)[0]
        
        # Get top predictions
        top3_idx = np.argsort(preds)[::-1][:3]
        top_predictions = [(class_names[i], float(preds[i])) for i in top3_idx]
        
        # Display results
        display_single_prediction_results(top_predictions)

def display_detection_details(index, detection):
    """Display detailed information for a single detection."""
    class_name = detection['class']
    confidence = detection['confidence']
    info = recycling_info.get(class_name, {})
    
    with st.expander(f"{info.get('icon', '❓')} Object {index + 1}: {class_name} ({confidence:.1%})", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Recycling Status:** {info.get('status', 'Unknown')}")
            st.markdown(f"**Guidance:** {info.get('notes', 'No information available.')}")
            st.markdown(f"**Disposal:** {info.get('disposal_guide', 'Check local guidelines.')}")
            
            # Audio feedback
            if st.session_state.get('enable_audio', True):
                speech_text = f"{class_name}. {info.get('status', '')}. {info.get('notes', '')}"
                speech_file = speak_text(speech_text)
                if speech_file:
                    st.audio(speech_file)
        
        with col2:
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "lightblue"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        # User feedback
        if st.session_state.get('enable_feedback', True):
            collect_user_feedback(class_name, confidence)

def display_detection_results(img, detections, display_col):
    """Display enhanced detection results."""
    with display_col:
        st.markdown("### 🎯 Detection Results")
        
        # Create annotated image
        img_annotated = create_detection_visualization(np.array(img), detections)
        st.image(img_annotated, use_column_width=True, caption="Detected Objects")
    
    # Detection summary
    st.markdown("### 📊 Detection Summary")
    
    # Create summary dataframe
    detection_data = []
    for i, detection in enumerate(detections):
        class_info = recycling_info.get(detection['class'], {})
        detection_data.append({
            'Object': i + 1,
            'Category': detection['class'],
            'Confidence': f"{detection['confidence']:.1%}",
            'Status': class_info.get('status', 'Unknown'),
            'Icon': class_info.get('icon', '❓')
        })
    
    detection_df = pd.DataFrame(detection_data)
    st.dataframe(detection_df, use_container_width=True)
    
    # Detailed information for each detection
    st.markdown("### 🔍 Detailed Analysis")
    for i, detection in enumerate(detections):
        display_detection_details(i, detection)

def process_uploaded_image(uploaded_file):
    """Process uploaded or camera image with enhanced analysis."""
    try:
        img = Image.open(uploaded_file).convert('RGB')
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Input Image")
            st.image(img, use_column_width=True, caption="Original Image")
        
        # Perform detection
        with st.spinner("🔍 Analyzing waste items..."):
            detections = waste_detector.detect_objects(img)
        
        # Update statistics
        st.session_state.app_usage_stats['total_predictions'] += 1
        st.session_state.app_usage_stats['total_detections'] += len(detections)
        
        if detections:
            display_detection_results(img, detections, col2)
        else:
            # Fallback to single classification
            perform_single_classification(img, col2)
            
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

def display_realtime_detection_sidebar():
    """Display real-time detection sidebar information."""
    st.markdown("### 📊 Live Detections")
    
    if st.session_state.realtime_detections:
        # Detection summary
        detection_df = pd.DataFrame(st.session_state.realtime_detections)
        
        if not detection_df.empty:
            # Summary statistics
            summary = detection_df.groupby('class').agg({
                'confidence': ['mean', 'count']
            }).round(3)
            summary.columns = ['Avg Confidence', 'Count']
            st.dataframe(summary, use_container_width=True)
            
            # Show recycling info for detected items
            unique_classes = detection_df['class'].unique()
            for class_name in unique_classes:
                if class_name in recycling_info:
                    info = recycling_info[class_name]
                    with st.expander(f"{info.get('icon', '❓')} {class_name}"):
                        st.write(f"**Status:** {info['status']}")
                        st.write(info['notes'])
                        
                        # Quick audio guide
                        if st.session_state.get('enable_audio', True) and st.button(f"Audio Guide", key=f"audio_{class_name}"):
                            speech_text = f"{class_name}. {info['status']}. {info['notes']}"
                            speech_file = speak_text(speech_text)
                            if speech_file:
                                st.audio(speech_file)
    else:
        st.info("👆 Point camera at waste items to see detections here.")
        
        # Show detection statistics
        stats = waste_detector.get_detection_stats()
        if stats:
            st.markdown("### 📈 Session Statistics")
            st.metric("Total Detections", stats['total_detections'])
            st.metric("Average Confidence", f"{stats['avg_confidence']:.1%}")
            
            if stats['most_common_class'][0] != 'None':
                st.metric("Most Common", stats['most_common_class'][0])

def display_realtime_detection():
    """Display real-time detection interface."""
    st.markdown("### 🎥 Real-time Waste Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        **Instructions:**
        - Allow camera access when prompted
        - Point camera at waste items
        - Detections will appear with bounding boxes
        - Recycling guidance provided in real-time
        """)
        
        # WebRTC streamer
        try:
            webrtc_ctx = webrtc_streamer(
                key="waste-detection",
                video_processor_factory=VideoProcessor,
                rtc_configuration=RTCConfiguration({
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            # Update detections from video processor
            if webrtc_ctx.video_processor:
                current_time = time.time()
                if current_time - st.session_state.last_detection_time > 2:
                    st.session_state.realtime_detections = webrtc_ctx.video_processor.last_detections
                    st.session_state.last_detection_time = current_time
                    
        except Exception as e:
            st.warning(f"⚠️ Camera access issue: {e}")
            st.info("Please ensure camera permissions are granted and try again.")
    
    with col2:
        display_realtime_detection_sidebar()

def display_analytics_dashboard():
    """Display comprehensive analytics dashboard."""
    st.markdown("### 📊 Waste Analytics Dashboard")
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", st.session_state.app_usage_stats['total_predictions'])
    with col2:
        st.metric("Total Detections", st.session_state.app_usage_stats['total_detections'])
    with col3:
        session_duration = datetime.datetime.now() - st.session_state.app_usage_stats['start_time']
        st.metric("Session Duration", f"{session_duration.seconds // 60} min")
    with col4:
        if st.session_state.prediction_history:
            avg_confidence = np.mean([
                float(h['confidence'].rstrip('%')) 
                for h in st.session_state.prediction_history 
                if 'confidence' in h
            ])
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    if st.session_state.prediction_history:
        # Prediction history chart
        st.markdown("### 📈 Prediction History")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Time series of predictions
        if 'timestamp' in history_df.columns:
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            daily_counts = history_df.groupby(history_df['timestamp'].dt.date).size()
            
            fig = px.line(
                x=daily_counts.index, 
                y=daily_counts.values,
                title="Daily Prediction Activity",
                labels={'x': 'Date', 'y': 'Number of Predictions'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Category distribution
        if 'predicted_class' in history_df.columns:
            category_dist = history_df['predicted_class'].value_counts()
            fig = px.pie(
                values=category_dist.values,
                names=category_dist.index,
                title="Waste Category Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        st.markdown("### 💾 Data Export")
        csv_data = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv_data,
            file_name=f"waste_analytics_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.info("📊 Analytics data will appear here after you make some predictions.")

# ============================
# MAIN APPLICATION LAYOUT
# ============================

# Enhanced UI Layout
st.markdown('<h1 class="main-header">♻️ AI Waste Classification & Recycling Assistant</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        Upload images, use real-time camera detection, or take photos to classify waste items 
        and get instant recycling guidance powered by deep learning.
    </p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("### 📸 Input Options")
    
    # Mode selection with icons
    mode = st.radio(
        "Select Input Mode:",
        ["📁 Upload Image", "📷 Take Photo", "🎥 Real-time Detection", "📊 Analytics"],
        index=0
    )
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        help="Adjust how confident the model needs to be for detection"
    )
    
    waste_detector.confidence_threshold = confidence_threshold
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        st.session_state.enable_audio = st.checkbox("Enable Audio Feedback", value=True)
        st.session_state.show_confidence = st.checkbox("Show Confidence Charts", value=True)
        st.session_state.enable_feedback = st.checkbox("Enable User Feedback", value=True)
    
    # Quick stats
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Predictions", st.session_state.app_usage_stats['total_predictions'])
    with col2:
        st.metric("Total Detections", st.session_state.app_usage_stats['total_detections'])
    
    # Clear button
    if st.button("🧹 Clear All", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ['session_id', 'app_usage_stats']:
                del st.session_state[key]
        initialize_session_state()
        st.rerun()

# Main content area based on mode
if mode == "📁 Upload Image":
    st.markdown("### 📁 Upload Waste Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload clear images of waste items for best results"
    )
    
    if uploaded_file is not None:
        process_uploaded_image(uploaded_file)

elif mode == "📷 Take Photo":
    st.markdown("### 📷 Capture Photo")
    camera_image = st.camera_input("Take a picture of waste item")
    if camera_image is not None:
        process_uploaded_image(camera_image)

elif mode == "🎥 Real-time Detection":
    display_realtime_detection()

elif mode == "📊 Analytics":
    display_analytics_dashboard()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>♻️ AI Waste Classification System | Built with Streamlit & TensorFlow</p>
    <p>Help make recycling smarter and more efficient! 🌍</p>
</div>
""", unsafe_allow_html=True)