import streamlit as st
import cv2
import numpy as np
from simple_detector import SimpleDrowsinessDetector
import time

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
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize all session state variables"""
    if "running" not in st.session_state:
        st.session_state.running = False
    if "eye_closed_counter" not in st.session_state:
        st.session_state.eye_closed_counter = 0
    if "mouth_open_counter" not in st.session_state:
        st.session_state.mouth_open_counter = 0
    if "drowsy_detected" not in st.session_state:
        st.session_state.drowsy_detected = False
    if "alert_active" not in st.session_state:
        st.session_state.alert_active = False
    if "frame_count" not in st.session_state:
        st.session_state.frame_count = 0
    if "last_frame_time" not in st.session_state:
        st.session_state.last_frame_time = None
    if "detector" not in st.session_state:
        st.session_state.detector = SimpleDrowsinessDetector()


def process_frame(img):
    """Process a single frame through the detector"""
    detector = st.session_state.detector

    # Flip frame for mirror effect
    img = cv2.flip(img, 1)

    # Run drowsiness detection
    processed_frame, drowsy_detected = detector.detect_drowsiness(img)

    # Update session state
    st.session_state.eye_closed_counter = detector.eye_closed_counter
    st.session_state.mouth_open_counter = detector.mouth_open_counter
    st.session_state.drowsy_detected = drowsy_detected
    st.session_state.alert_active = detector.alert_active
    st.session_state.frame_count += 1

    # Convert BGR to RGB for Streamlit
    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

    return processed_rgb


def main():
    # Initialize session state
    init_session_state()

    # Header
    st.markdown('<p class="main-header">🚨 Drowsiness Detection System</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("**Detection Thresholds:**")
        st.info("- Eyes closed for 1.5+ seconds triggers alert")
        st.info("- Yawning for 0.5+ seconds triggers alert")

        st.markdown("---")
        st.header("📋 Instructions")
        st.markdown("""
        1. Click 'Start Detection' to begin
        2. Allow camera access when prompted
        3. Position your face clearly in front of camera
        4. Keep eyes visible for accurate detection
        5. Visual alert appears when drowsiness detected
        """)

        st.markdown("---")

        # Start/Stop buttons
        if not st.session_state.running:
            if st.button("▶️ Start Detection", type="primary"):
                st.session_state.running = True
                st.session_state.detector = SimpleDrowsinessDetector()
                st.rerun()
        else:
            if st.button("⏹️ Stop Detection", type="secondary"):
                st.session_state.running = False
                st.session_state.detector.stop_alert()
                st.rerun()

        st.markdown("---")

        # Reset button
        if st.button("🔄 Reset Detector"):
            st.session_state.detector.reset()
            st.session_state.eye_closed_counter = 0
            st.session_state.mouth_open_counter = 0
            st.session_state.frame_count = 0
            st.success("Detector reset!")
            st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📹 Live Feed")

        if not st.session_state.running:
            st.info("Click 'Start Detection' to begin monitoring")
            st.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Camera off", use_container_width=True)
        else:
            # Use st.camera_input for live feed
            # Note: This captures frames when user interacts or we can use session state tricks
            cam_frame = st.camera_input("Camera", key="camera_feed")

            if cam_frame is not None:
                # Read the frame
                img = np.array(cam_frame)
                processed = process_frame(img)
                st.image(processed, use_container_width=True)

                # Auto-refresh for continuous feed
                st.rerun()
            else:
                st.info("Waiting for camera...")

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

        # Update status display from session state
        eye_seconds = st.session_state.eye_closed_counter / 30.0
        mouth_seconds = st.session_state.mouth_open_counter / 30.0

        # Status box
        if st.session_state.drowsy_detected:
            status_placeholder.markdown(
                '<div class="status-box alert-status">🚨 DROWSINESS DETECTED!</div>',
                unsafe_allow_html=True
            )
            alert_placeholder.warning("⚠️ Visual alert is ACTIVE")
        elif st.session_state.eye_closed_counter > 15 or st.session_state.mouth_open_counter > 5:
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
            delta=f"Threshold: 1.5s" if st.session_state.eye_closed_counter > 15 else None
        )
        mouth_placeholder.metric(
            "Yawning Duration",
            f"{mouth_seconds:.1f}s",
            delta=f"Threshold: 0.5s" if st.session_state.mouth_open_counter > 5 else None
        )

        # Detection info
        info_placeholder.markdown(f"""
        - **Frame:** {st.session_state.frame_count}
        - **Eye Status:** {'Closed' if st.session_state.eye_closed_counter > 0 else 'Open'}
        - **Mouth Status:** {'Open' if st.session_state.mouth_open_counter > 0 else 'Closed'}
        """)


if __name__ == "__main__":
    main()
