import streamlit as st
import cv2
import av
from simple_detector import SimpleDrowsinessDetector
from streamlit_webrtc import WebRtcMode, webrtc_streamer

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


# Use session state directly for shared metrics (not cached)
def init_shared_state():
    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "eye_closed_counter": 0,
            "mouth_open_counter": 0,
            "drowsy_detected": False,
            "alert_active": False,
            "frame_count": 0
        }
    return st.session_state.metrics


class DrowsinessVideoProcessor:
    """Video processor that runs drowsiness detection on each frame"""

    def __init__(self, metrics):
        self.detector = SimpleDrowsinessDetector()
        self.metrics = metrics

    def recv(self, frame):
        """Process incoming video frame"""
        # Convert AVFrame to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Flip frame for mirror effect
        img = cv2.flip(img, 1)

        # Run drowsiness detection
        processed_frame, drowsy_detected = self.detector.detect_drowsiness(img)

        # Update shared metrics for UI
        self.metrics["eye_closed_counter"] = self.detector.eye_closed_counter
        self.metrics["mouth_open_counter"] = self.detector.mouth_open_counter
        self.metrics["drowsy_detected"] = drowsy_detected
        self.metrics["alert_active"] = self.detector.alert_active
        self.metrics["frame_count"] = self.metrics.get("frame_count", 0) + 1

        # Convert back to RGB for WebRTC
        processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Convert numpy array back to AVFrame
        return av.VideoFrame.from_ndarray(processed_rgb, format="rgb24")


def main():
    # Header
    st.markdown('<p class="main-header">🚨 Drowsiness Detection System</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("**Detection Thresholds:**")
        st.info("- Eyes closed for 2+ seconds triggers alert")
        st.info("- Yawning for 1+ seconds triggers alert")

        st.markdown("---")
        st.header("📋 Instructions")
        st.markdown("""
        1. Allow camera access when prompted
        2. Position your face clearly in front of camera
        3. The system will detect:
           - Eye closure (drowsiness)
           - Yawning (fatigue)
        4. Visual alert appears when drowsiness detected
        """)

        st.markdown("---")
        # Reset button
        if st.button("🔄 Reset Detector"):
            if 'processor' in st.session_state:
                st.session_state.processor.detector.reset()
                st.session_state.metrics["eye_closed_counter"] = 0
                st.session_state.metrics["mouth_open_counter"] = 0
                st.session_state.metrics["frame_count"] = 0
                st.success("Detector reset!")
                st.rerun()

    # Initialize shared metrics
    metrics = init_shared_state()

    # Create processor - it will be used when the WebRTC stream starts
    if 'processor' not in st.session_state:
        st.session_state.processor = DrowsinessVideoProcessor(metrics)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📹 Live Feed")
        # WebRTC streamer with video processor - starts automatically when user allows camera
        webrtc_streamer(
            key="drowsy-detection",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True},
            video_processor_factory=lambda: DrowsinessVideoProcessor(metrics),
            async_processing=True,
        )

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

        # Update status display from shared metrics
        eye_seconds = metrics["eye_closed_counter"] / 30.0
        mouth_seconds = metrics["mouth_open_counter"] / 30.0

        # Status box
        if metrics["drowsy_detected"]:
            status_placeholder.markdown(
                '<div class="status-box alert-status">🚨 DROWSINESS DETECTED!</div>',
                unsafe_allow_html=True
            )
            alert_placeholder.warning("⚠️ Visual alert is ACTIVE")
        elif metrics["eye_closed_counter"] > 10 or metrics["mouth_open_counter"] > 5:
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
            delta=f"Threshold: 2.0s" if metrics["eye_closed_counter"] > 10 else None
        )
        mouth_placeholder.metric(
            "Yawning Duration",
            f"{mouth_seconds:.1f}s",
            delta=f"Threshold: 1.0s" if metrics["mouth_open_counter"] > 5 else None
        )

        # Detection info
        info_placeholder.markdown(f"""
        - **Frame:** {metrics.get('frame_count', 0)}
        - **Eye Status:** {'Closed' if metrics['eye_closed_counter'] > 0 else 'Open'}
        - **Mouth Status:** {'Open' if metrics['mouth_open_counter'] > 0 else 'Closed'}
        """)


if __name__ == "__main__":
    main()
