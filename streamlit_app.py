import streamlit as st
import cv2
import numpy as np
from simple_detector import SimpleDrowsinessDetector

# Page configuration
st.set_page_config(
    page_title="Drowsiness Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .alert-status {
        background-color: #FFEBEE;
        border: 3px solid #F44336;
        color: #C62828;
    }
    .safe-status {
        background-color: #E8F5E9;
        border: 3px solid #4CAF50;
        color: #2E7D32;
    }
    .warning-status {
        background-color: #FFF3E0;
        border: 3px solid #FF9800;
        color: #E65100;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize detector once
@st.cache_resource
def get_detector():
    return SimpleDrowsinessDetector()


def process_frame(frame, detector):
    """Process a single frame and return results"""
    processed_frame, drowsy_detected = detector.detect_drowsiness(frame)
    return processed_frame, drowsy_detected, detector.eye_closed_counter, detector.mouth_open_counter


def main():
    # Header
    st.markdown('<p class="main-header">🚨 Drowsiness Detection System</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("**Detection Thresholds:**")
        st.info("- Eyes closed for 2+ seconds triggers alert")
        st.info("- Yawning for 0.5+ seconds triggers alert")

        st.markdown("---")
        st.header("📋 Instructions")
        st.markdown("""
        1. Allow camera access when prompted
        2. Position your face clearly in front of camera
        3. The system will detect:
           - Eye closure (drowsiness)
           - Yawning (fatigue)
        4. Alert will sound when drowsiness detected
        """)

        st.markdown("---")
        if st.button("🔄 Reset Detector"):
            if 'detector' in st.session_state:
                st.session_state.detector.reset()
                st.success("Detector reset!")

    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = get_detector()

    if 'running' not in st.session_state:
        st.session_state.running = False

    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📹 Live Feed")
        video_placeholder = st.empty()

    with col2:
        st.header("📊 Status")
        status_placeholder = st.empty()
        eyes_placeholder = st.empty()
        mouth_placeholder = st.empty()
        alert_placeholder = st.empty()

        # Info box
        st.markdown("---")
        st.header("ℹ️ Detection Info")
        info_placeholder = st.empty()

    # Camera capture using st.camera_input
    camera_frame = st.camera_input("Camera")

    # Control buttons - placed after camera input for better UX
    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶️ Start Detection"):
            st.session_state.running = True
            st.rerun()
    with col_btn2:
        if st.button("⏹️ Stop Detection"):
            st.session_state.running = False
            st.session_state.detector.stop_alert()
            st.rerun()

    # Show status based on running state
    if not st.session_state.running:
        video_placeholder.info("Click 'Start Detection' to begin monitoring")
        status_placeholder.markdown(
            '<div class="status-box safe-status">⏸️ Paused</div>',
            unsafe_allow_html=True
        )
        st.stop()

    if camera_frame is None:
        video_placeholder.warning("📷 Please allow camera access to start detection")
        st.stop()

    # Read image from camera
    image = np.array(camera_frame)

    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Process frame
    processed_frame, drowsy_detected, eye_counter, mouth_counter = process_frame(
        frame, st.session_state.detector
    )

    st.session_state.frame_count += 1

    # Convert BGR to RGB for Streamlit
    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

    # Display video
    video_placeholder.image(processed_frame_rgb, width=640)

    # Update status display
    eye_seconds = eye_counter / 30.0
    mouth_seconds = mouth_counter / 30.0

    # Status box
    if drowsy_detected:
        status_placeholder.markdown(
            '<div class="status-box alert-status">🚨 DROWSINESS DETECTED!</div>',
            unsafe_allow_html=True
        )
        alert_placeholder.warning("⚠️ Alert is ACTIVE")
    elif eye_counter > 10 or mouth_counter > 5:
        status_placeholder.markdown(
            '<div class="status-box warning-status">⚠️ Warning - Possible Fatigue</div>',
            unsafe_allow_html=True
        )
        alert_placeholder.info("Monitoring...")
    else:
        status_placeholder.markdown(
            '<div class="status-box safe-status">✅ Driver is Alert</div>',
            unsafe_allow_html=True
        )
        alert_placeholder.success("No alert active")

    # Metrics
    eyes_placeholder.metric(
        "Eyes Closed Duration",
        f"{eye_seconds:.1f}s",
        delta=f"Threshold: 2.0s" if eye_counter > 10 else None
    )
    mouth_placeholder.metric(
        "Yawning Duration",
        f"{mouth_seconds:.1f}s",
        delta=f"Threshold: 0.5s" if mouth_counter > 5 else None
    )

    # Detection info
    info_placeholder.markdown(f"""
    - **Frame:** {st.session_state.frame_count}
    - **Eye Status:** {'Closed' if eye_counter > 0 else 'Open'}
    - **Mouth Status:** {'Open (Yawning)' if mouth_counter > 0 else 'Closed'}
    """)


if __name__ == "__main__":
    main()
