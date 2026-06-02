# # app.py
# # Run with: streamlit run app.py

# import streamlit as st
# import cv2
# import numpy as np
# import time
# import tempfile
# import os
# from pipeline import (
#     DepthEstimator, TTCCalculator,
#     detect_conditions, draw_dashboard, SoundAlert
# )

# # ── Page config ────────────────────────────────────
# st.set_page_config(
#     page_title = "VisionGuard",
#     page_icon  = "🛡️",
#     layout     = "wide"
# )

# # ── Custom CSS ─────────────────────────────────────
# st.markdown("""
# <style>
#     .main { background-color: #0F172A; }
#     .stApp { background-color: #0F172A; }
#     h1 { color: #60A5FA !important; }
#     h2 { color: #93C5FD !important; }
#     h3 { color: #BFDBFE !important; }
#     .metric-card {
#         background: #1E293B;
#         border-radius: 10px;
#         padding: 15px;
#         margin: 5px;
#         border: 1px solid #334155;
#     }
#     .status-ok  { color: #22c55e; font-weight: bold; }
#     .status-warn{ color: #f97316; font-weight: bold; }
#     .status-crit{ color: #ef4444; font-weight: bold; }
# </style>
# """, unsafe_allow_html=True)

# # ── Header ─────────────────────────────────────────
# st.markdown("""
# <h1 style='text-align:center; font-size:3em;'>
#     🛡️ VisionGuard
# </h1>
# <p style='text-align:center; color:#94A3B8; font-size:1.2em;'>
#     Real-Time Depth-Aware Object Detection and Risk Analysis
# </p>
# <p style='text-align:center; color:#64748B;'>
#     Camera-Only ADAS for Indian Roads — No LiDAR Required
# </p>
# <hr style='border-color:#334155;'>
# """, unsafe_allow_html=True)

# # ── Load models (cached so they load only once) ────
# @st.cache_resource
# def load_models():
#     from ultralytics import YOLO
#     from stable_baselines3 import PPO

#     st.info("Loading models — please wait...")
#     yolo  = YOLO('models/visionguard_best.pt')
#     rl = PPO.load('models/alert_policy_local')
#     depth = DepthEstimator()
#     return yolo, rl, depth

# # ── Sidebar ────────────────────────────────────────
# with st.sidebar:
#     st.markdown("## ⚙️ System Info")
#     st.markdown("""
#     <div class='metric-card'>
#         <p class='status-ok'>✅ YOLO Model: Loaded</p>
#         <p style='color:#94A3B8; font-size:0.85em;'>
#             mAP50: 0.777 | Classes: 10
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("""
#     <div class='metric-card'>
#         <p class='status-ok'>✅ Depth Model: Loaded</p>
#         <p style='color:#94A3B8; font-size:0.85em;'>
#             DepthAnything v2 Small
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("""
#     <div class='metric-card'>
#         <p class='status-ok'>✅ RL Agent: Loaded</p>
#         <p style='color:#94A3B8; font-size:0.85em;'>
#             PPO | 200,000 steps trained
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("---")
#     st.markdown("## 🎛️ Settings")
#     conf_thresh  = st.slider("Detection Confidence", 0.3, 0.8, 0.4, 0.05)
#     depth_every  = st.slider("Depth Frequency (every N frames)", 1, 5, 3)
#     enable_sound = st.checkbox("Enable Sound Alerts", value=True)
#     show_depth   = st.checkbox("Show Depth Map", value=False)

#     st.markdown("---")
#     st.markdown("## 📊 Model Performance")
#     st.markdown("""
#     | Class | AP50 |
#     |-------|------|
#     | bike | 0.995 |
#     | autorickshaw | 0.984 |
#     | car | 0.967 |
#     | bus | 0.950 |
#     | bicycle | 0.888 |
#     | motorbike | 0.800 |
#     """)

# # ── Main content ───────────────────────────────────
# tab1, tab2, tab3 = st.tabs([
#     "🎥 Live Demo",
#     "📊 Results",
#     "ℹ️ About"
# ])

# # ══ TAB 1: LIVE DEMO ══════════════════════════════
# with tab1:
#     st.markdown("### Upload Dashcam Video")
#     video_file = st.file_uploader(
#         "Choose a dashcam video file",
#         type=['mp4', 'avi', 'mov'],
#         help="Upload any driving footage to analyse"
#     )

#     col1, col2, col3, col4 = st.columns(4)
#     risk_placeholder   = col1.empty()
#     objects_placeholder= col2.empty()
#     ttc_placeholder    = col3.empty()
#     cond_placeholder   = col4.empty()

#     # Display metric cards
#     risk_placeholder.metric("🔴 Risk Score", "—")
#     objects_placeholder.metric("📦 Objects", "—")
#     ttc_placeholder.metric("⏱️ Min TTC", "—")
#     cond_placeholder.metric("🌤️ Conditions", "—")

#     if video_file is not None:
#         # Save uploaded video to temp file
#         tfile = tempfile.NamedTemporaryFile(
#             delete=False, suffix='.mp4'
#         )
#         tfile.write(video_file.read())
#         tfile.close()

#         # Load models
#         with st.spinner("Loading AI models..."):
#             yolo_model, rl_agent, depth_estimator = load_models()
#         st.success("All models loaded!")

#         # Two columns: video output + depth map
#         if show_depth:
#             vid_col, depth_col = st.columns([2, 1])
#             stframe_depth = depth_col.empty()
#             depth_col.caption("Depth Map")
#         else:
#             vid_col = st.container()

#         stframe = vid_col.empty()

#         # Stop button
#         stop_btn = st.button("⏹️ Stop Processing")

#         # Initialise components
#         ttc_calc   = TTCCalculator(fps=30)
#         sound      = SoundAlert()
#         cap        = cv2.VideoCapture(tfile.name)
#         fps        = int(cap.get(cv2.CAP_PROP_FPS)) or 30
#         frame_num  = 0
#         depth_raw  = None

#         progress = st.progress(0)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         while cap.isOpened() and not stop_btn:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             t0 = time.time()

#             # Resize for speed
#             frame = cv2.resize(frame, (640, 480))

#             # Conditions
#             conditions   = detect_conditions(frame)
#             weather_code = 3 if 'NIGHT' in conditions else \
#                            2 if 'FOG'   in conditions else \
#                            1 if 'RAIN'  in conditions else 0

#             # Depth (every N frames)
#             if frame_num % depth_every == 0:
#                 depth_raw, depth_colored = \
#                     depth_estimator.get_depth_map(frame)

#             # YOLO detection
#             yolo_out   = yolo_model(
#                 frame, verbose=False, conf=conf_thresh
#             )
#             boxes      = yolo_out[0].boxes
#             active_ids = []
#             detections = []
#             max_risk   = 0

#             for i, box in enumerate(boxes):
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cls_id   = int(box.cls[0])
#                 cls_name = yolo_model.names[cls_id]
#                 active_ids.append(i)

#                 # Distance
#                 dist = depth_estimator.get_distance(
#                     depth_raw, (x1,y1,x2,y2)
#                 ) if depth_raw is not None else 20.0

#                 # TTC
#                 ttc = ttc_calc.update(i, dist)

#                 # RL alert
#                 ttc_cap = min(ttc, 30) if ttc < 999 else 30
#                 obs = np.array(
#                     [dist, ttc_cap, float(cls_id),
#                      float(weather_code), 0.8],
#                     dtype=np.float32
#                 )
#                 alert, _ = rl_agent.predict(
#                     obs, deterministic=True
#                 )

#                 # Risk score
#                 ds = 95 if dist<5 else 75 if dist<10 else \
#                      55 if dist<15 else 35 if dist<25 else 10
#                 ts = 95 if ttc_cap<2 else 65 if ttc_cap<4 else \
#                      35 if ttc_cap<7 else 10
#                 risk    = min(100, round(ds*0.35 + ts*0.65))
#                 max_risk= max(max_risk, risk)

#                 detections.append({
#                     'bbox':     (x1,y1,x2,y2),
#                     'class':    cls_name,
#                     'distance': dist,
#                     'ttc':      ttc_cap,
#                     'alert':    int(alert),
#                     'risk':     risk
#                 })

#             ttc_calc.cleanup(active_ids)

#             # Draw overlay
#             fps_actual = 1.0 / max(time.time()-t0, 0.001)
#             frame, highest_alert = draw_dashboard(
#                 frame, detections, conditions,
#                 max_risk, fps_actual
#             )

#             # Sound alert
#             if enable_sound:
#                 sound.play(highest_alert)

#             # Show in Streamlit
#             stframe.image(
#                 cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
#                 channels="RGB",
#                 use_container_width=True
#             )

#             if show_depth and depth_raw is not None:
#                 stframe_depth.image(
#                     cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB),
#                     channels="RGB",
#                     use_container_width=True
#                 )

#             # Update metric cards
#             min_ttc = min(
#                 (d['ttc'] for d in detections if d['ttc'] < 30),
#                 default=999
#             )
#             ttc_str = f"{min_ttc}s" if min_ttc < 30 else "Safe"

#             risk_placeholder.metric("🔴 Risk Score", f"{max_risk}/100")
#             objects_placeholder.metric("📦 Objects", len(detections))
#             ttc_placeholder.metric("⏱️ Min TTC", ttc_str)
#             cond_placeholder.metric(
#                 "🌤️ Conditions", conditions[0]
#             )

#             frame_num += 1
#             if total_frames > 0:
#                 progress.progress(
#                     min(frame_num / total_frames, 1.0)
#                 )

#         cap.release()
#         os.unlink(tfile.name)
#         st.success("Processing complete!")

# # ══ TAB 2: RESULTS ════════════════════════════════
# with tab2:
#     st.markdown("### Model Performance Results")

#     col1, col2, col3 = st.columns(3)
#     col1.metric("Overall mAP50",   "0.777", "+0.35 vs baseline")
#     col2.metric("Training Images", "3,438", "10 classes")
#     col3.metric("RL Training",     "200K",  "PPO steps")

#     st.markdown("---")
#     st.markdown("#### Per-Class Detection Accuracy")

#     import pandas as pd
#     results_df = pd.DataFrame({
#         'Class':    ['bike','autorickshaw','car','bus',
#                      'bicycle','cycle','motorbike',
#                      'person','tractor'],
#         'AP50':     [0.995, 0.984, 0.967, 0.950,
#                      0.888, 0.862, 0.800, 0.052, 0.492],
#         'Grade':    ['Excellent','Excellent','Excellent',
#                      'Excellent','Excellent','Excellent',
#                      'Good','Limited','Limited']
#     })
#     st.dataframe(
#         results_df,
#         use_container_width=True,
#         hide_index=True
#     )

#     st.markdown("---")
#     st.markdown("#### System Comparison")
#     comp_df = pd.DataFrame({
#         'System':   ['Tesla ADAS','Mobileye','VisionGuard'],
#         'Hardware': ['LiDAR+Radar+Camera',
#                      'Proprietary Camera+Chip',
#                      'Single RGB Camera Only'],
#         'Cost':     ['$10,000+','$1,000+','~$0'],
#         'Indian Road Classes': ['No','No','Yes']
#     })
#     st.dataframe(comp_df, use_container_width=True,
#                  hide_index=True)

# # ══ TAB 3: ABOUT ══════════════════════════════════
# with tab3:
#     st.markdown("""
#     ### About VisionGuard

#     VisionGuard is a camera-only Advanced Driver Assistance
#     System built for Indian road conditions as a BTech
#     Deep Learning project.

#     #### Pipeline Modules
#     1. **YOLOv8** — Fine-tuned object detector (mAP50: 0.777)
#     2. **DepthAnything v2** — Monocular depth estimation
#     3. **TTC Calculator** — Time-to-collision via regression
#     4. **Conditions Detector** — Night/Rain/Fog detection
#     5. **PPO RL Agent** — Intelligent alert policy

#     #### Key Innovation
#     Replaces expensive LiDAR sensors (USD 4,000-75,000)
#     with a single RGB camera using deep learning — making
#     ADAS accessible to every vehicle owner.

#     #### Alert Levels
#     | Level | Trigger | Action |
#     |-------|---------|--------|
#     | SAFE | TTC > 10s | Silent |
#     | CAUTION | TTC 6-10s | Soft beep |
#     | WARNING | TTC 2.5-6s | Urgent beep |
#     | BRAKE NOW | TTC < 2.5s | Alarm |
#     """)





# app.py
# Run with: C:\Users\Tanya\AppData\Local\Programs\Python\Python312\python.exe -m streamlit run app.py

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from pipeline import (
    DepthEstimator, TTCCalculator,
    detect_conditions, draw_dashboard, SoundAlert
)

# ── Page config ────────────────────────────────────
st.set_page_config(
    page_title = "VisionGuard",
    page_icon  = "🛡️",
    layout     = "wide"
)

# ── Custom CSS ─────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0F172A; }
    .stApp { background-color: #0F172A; }
    h1 { color: #60A5FA !important; }
    h2 { color: #93C5FD !important; }
    h3 { color: #BFDBFE !important; }
    .metric-card {
        background: #1E293B;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        border: 1px solid #334155;
    }
    .status-ok   { color: #22c55e; font-weight: bold; }
    .status-warn { color: #f97316; font-weight: bold; }
    .status-crit { color: #ef4444; font-weight: bold; }
    .phone-box {
        background: #1E293B;
        border: 2px solid #3B82F6;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .step-badge {
        background: #3B82F6;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; font-size:3em;'>
    🛡️ VisionGuard
</h1>
<p style='text-align:center; color:#94A3B8; font-size:1.2em;'>
    Real-Time Depth-Aware Object Detection and Risk Analysis
</p>
<p style='text-align:center; color:#64748B;'>
    Camera-Only ADAS for Indian Roads — No LiDAR Required
</p>
<hr style='border-color:#334155;'>
""", unsafe_allow_html=True)

# ── Load models (cached so they load only once) ────
@st.cache_resource
def load_models():
    from ultralytics import YOLO
    from stable_baselines3 import PPO
    st.info("Loading models — please wait...")
    yolo  = YOLO('models/visionguard_best.pt')
    rl    = PPO.load('models/alert_policy_local')
    depth = DepthEstimator()
    return yolo, rl, depth

# ── Sidebar ────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ System Info")
    st.markdown("""
    <div class='metric-card'>
        <p class='status-ok'>✅ YOLO Model: Loaded</p>
        <p style='color:#94A3B8; font-size:0.85em;'>
            mAP50: 0.777 | Classes: 10
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='metric-card'>
        <p class='status-ok'>✅ Depth Model: Loaded</p>
        <p style='color:#94A3B8; font-size:0.85em;'>
            DepthAnything v2 Small
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='metric-card'>
        <p class='status-ok'>✅ RL Agent: Loaded</p>
        <p style='color:#94A3B8; font-size:0.85em;'>
            PPO | 200,000 steps trained
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🎛️ Settings")
    conf_thresh  = st.slider("Detection Confidence", 0.3, 0.8, 0.4, 0.05)
    depth_every  = st.slider("Depth Frequency (every N frames)", 1, 5, 3)
    enable_sound = st.checkbox("Enable Sound Alerts", value=True)
    show_depth   = st.checkbox("Show Depth Map", value=False)

    st.markdown("---")
    st.markdown("## 📊 Model Performance")
    st.markdown("""
    | Class | AP50 |
    |-------|------|
    | bike | 0.995 |
    | autorickshaw | 0.984 |
    | car | 0.967 |
    | bus | 0.950 |
    | bicycle | 0.888 |
    | motorbike | 0.800 |
    """)

# ── Main content ───────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🎥 Live Demo",
    "📊 Results",
    "ℹ️ About"
])

# ══ TAB 1: LIVE DEMO ══════════════════════════════
with tab1:

    # ── Input source selector ──────────────────────
    st.markdown("### Choose Input Source")
    input_mode = st.radio(
        "Select how you want to provide video input:",
        [
            "📁 Upload Video File",
            "📱 Phone Camera (IP Webcam)"
        ],
        horizontal=True
    )

    st.markdown("---")

    # ══ OPTION 1: Upload Video File (original) ══
    if input_mode == "📁 Upload Video File":

        st.markdown("### Upload Dashcam Video")
        video_file = st.file_uploader(
            "Choose a dashcam video file",
            type=['mp4', 'avi', 'mov'],
            help="Upload any driving footage to analyse"
        )

        col1, col2, col3, col4 = st.columns(4)
        risk_placeholder    = col1.empty()
        objects_placeholder = col2.empty()
        ttc_placeholder     = col3.empty()
        cond_placeholder    = col4.empty()

        risk_placeholder.metric("🔴 Risk Score", "—")
        objects_placeholder.metric("📦 Objects",  "—")
        ttc_placeholder.metric("⏱️ Min TTC",      "—")
        cond_placeholder.metric("🌤️ Conditions",  "—")

        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            tfile.close()

            with st.spinner("Loading AI models..."):
                yolo_model, rl_agent, depth_estimator = load_models()
            st.success("All models loaded!")

            if show_depth:
                vid_col, depth_col = st.columns([2, 1])
                stframe_depth = depth_col.empty()
                depth_col.caption("Depth Map")
            else:
                vid_col = st.container()

            stframe  = vid_col.empty()
            stop_btn = st.button("⏹️ Stop Processing")

            ttc_calc     = TTCCalculator(fps=30)
            sound        = SoundAlert()
            cap          = cv2.VideoCapture(tfile.name)
            fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            frame_num    = 0
            depth_raw    = None
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress     = st.progress(0)

            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.time()
                frame = cv2.resize(frame, (640, 480))

                conditions   = detect_conditions(frame)
                weather_code = 3 if 'NIGHT' in conditions else \
                               2 if 'FOG'   in conditions else \
                               1 if 'RAIN'  in conditions else 0

                if frame_num % depth_every == 0:
                    depth_raw, depth_colored = depth_estimator.get_depth_map(frame)

                yolo_out   = yolo_model(frame, verbose=False, conf=conf_thresh)
                boxes      = yolo_out[0].boxes
                active_ids = []
                detections = []
                max_risk   = 0

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id   = int(box.cls[0])
                    cls_name = yolo_model.names[cls_id]
                    active_ids.append(i)

                    dist    = depth_estimator.get_distance(
                        depth_raw, (x1, y1, x2, y2)
                    ) if depth_raw is not None else 20.0
                    ttc     = ttc_calc.update(i, dist)
                    ttc_cap = min(ttc, 30) if ttc < 999 else 30
                    obs     = np.array(
                        [dist, ttc_cap, float(cls_id), float(weather_code), 0.8],
                        dtype=np.float32
                    )
                    alert, _ = rl_agent.predict(obs, deterministic=True)
                    ds   = 95 if dist<5 else 75 if dist<10 else \
                           55 if dist<15 else 35 if dist<25 else 10
                    ts   = 95 if ttc_cap<2 else 65 if ttc_cap<4 else \
                           35 if ttc_cap<7 else 10
                    risk     = min(100, round(ds*0.35 + ts*0.65))
                    max_risk = max(max_risk, risk)
                    detections.append({
                        'bbox': (x1, y1, x2, y2), 'class': cls_name,
                        'distance': dist, 'ttc': ttc_cap,
                        'alert': int(alert), 'risk': risk
                    })

                ttc_calc.cleanup(active_ids)
                fps_actual = 1.0 / max(time.time()-t0, 0.001)
                frame, highest_alert = draw_dashboard(
                    frame, detections, conditions, max_risk, fps_actual
                )
                if enable_sound:
                    sound.play(highest_alert)

                stframe.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_container_width=True
                )
                if show_depth and depth_raw is not None:
                    stframe_depth.image(
                        cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB),
                        channels="RGB", use_container_width=True
                    )

                min_ttc = min(
                    (d['ttc'] for d in detections if d['ttc'] < 30), default=999
                )
                ttc_str = f"{min_ttc}s" if min_ttc < 30 else "Safe"
                risk_placeholder.metric("🔴 Risk Score",  f"{max_risk}/100")
                objects_placeholder.metric("📦 Objects",  len(detections))
                ttc_placeholder.metric("⏱️ Min TTC",      ttc_str)
                cond_placeholder.metric("🌤️ Conditions",  conditions[0])

                frame_num += 1
                if total_frames > 0:
                    progress.progress(min(frame_num / total_frames, 1.0))

            cap.release()
            os.unlink(tfile.name)
            st.success("Processing complete!")

    # ══ OPTION 2: Phone Camera via IP Webcam ══
    elif input_mode == "📱 Phone Camera (IP Webcam)":

        # ── Setup instructions ──
        st.markdown("""
        <div class='phone-box'>
            <h3 style='color:#60A5FA; margin-top:0;'>
                📱 How to Connect Your Phone Camera
            </h3>
            <p style='color:#94A3B8;'>
                Follow these steps before clicking Start:
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
            **On Your Phone:**

            **1.** Install **IP Webcam** app
            (by Pavel Khlebovich — free on Play Store)

            **2.** Open the app

            **3.** Scroll down → tap **"Start server"**

            **4.** Note the IP address shown at bottom
            Example: `http://192.168.1.5:8080`

            **5.** Point phone camera at traffic/road
            """)

        with col_b:
            st.markdown("""
            **On Your Laptop:**

            **1.** Make sure phone & laptop are on
            **same WiFi network**

            **2.** Type the IP address below

            **3.** Click **"Connect & Start"**

            **4.** Pipeline runs live on phone feed!

            **Tip:** Keep phone plugged in while demoing
            """)

        st.markdown("---")

        # ── IP input ──
        st.markdown("### Enter Phone IP Address")
        phone_ip = st.text_input(
            "Phone IP from IP Webcam app:",
            placeholder="http://192.168.1.5:8080",
            help="Copy the exact address shown in IP Webcam app"
        )

        # ── Test connection button ──
        col_test, col_start, col_stop = st.columns([1, 1, 1])

        test_btn  = col_test.button("🔍 Test Connection")
        start_btn = col_start.button("▶️ Connect & Start Live Feed")

        # Test connection
        if test_btn:
            if not phone_ip:
                st.error("Please enter IP address first!")
            else:
                with st.spinner("Testing connection to phone..."):
                    stream_url = phone_ip.rstrip('/') + "/video"
                    test_cap   = cv2.VideoCapture(stream_url)
                    time.sleep(2)
                    if test_cap.isOpened():
                        ret, _ = test_cap.read()
                        test_cap.release()
                        if ret:
                            st.success(
                                f"✅ Connected to phone camera at {phone_ip}"
                            )
                        else:
                            st.warning(
                                "Connected but no frames received. "
                                "Check if server is running on phone."
                            )
                    else:
                        st.error(
                            "❌ Cannot connect. Check: "
                            "1) Same WiFi? "
                            "2) IP Webcam server running? "
                            "3) IP address correct?"
                        )

        # ── Live feed ──
        if start_btn:
            if not phone_ip:
                st.error("Please enter your phone IP address first!")
                st.stop()

            stream_url = phone_ip.rstrip('/') + "/video"

            with st.spinner("Loading AI models..."):
                yolo_model, rl_agent, depth_estimator = load_models()

            # Connect to phone
            cap = cv2.VideoCapture(stream_url)
            time.sleep(1)

            if not cap.isOpened():
                st.error(
                    "Cannot connect to phone camera! "
                    "Check IP address and make sure "
                    "IP Webcam server is running."
                )
                st.stop()

            st.success(f"📱 Connected to phone at {phone_ip}")
            st.info(
                "Point your phone camera at traffic. "
                "Detection runs live!"
            )

            # ── Live metrics ──
            col1, col2, col3, col4 = st.columns(4)
            risk_ph = col1.empty()
            obj_ph  = col2.empty()
            ttc_ph  = col3.empty()
            cond_ph = col4.empty()

            risk_ph.metric("🔴 Risk Score", "—")
            obj_ph.metric("📦 Objects",     "—")
            ttc_ph.metric("⏱️ Min TTC",     "—")
            cond_ph.metric("🌤️ Conditions", "—")

            # Video display
            if show_depth:
                vid_col, depth_col = st.columns([2, 1])
                stframe_depth = depth_col.empty()
                depth_col.caption("Depth Map")
            else:
                vid_col = st.container()

            stframe  = vid_col.empty()
            stop_btn = st.button("⏹️ Stop Live Feed")

            st.markdown("""
            <p style='color:#64748B; font-size:0.85em;'>
            💡 Live feed from phone camera.
            Processing speed depends on your WiFi and laptop.
            </p>
            """, unsafe_allow_html=True)

            # ── Processing loop ──
            ttc_calc  = TTCCalculator(fps=30)
            sound     = SoundAlert()
            frame_num = 0
            depth_raw = None

            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    st.warning(
                        "Lost connection to phone. "
                        "Check WiFi and restart server."
                    )
                    break

                t0    = time.time()
                frame = cv2.resize(frame, (640, 480))

                conditions   = detect_conditions(frame)
                weather_code = 3 if 'NIGHT' in conditions else \
                               2 if 'FOG'   in conditions else \
                               1 if 'RAIN'  in conditions else 0

                # Depth every N frames
                if frame_num % depth_every == 0:
                    depth_raw, depth_colored = \
                        depth_estimator.get_depth_map(frame)

                # YOLO detection
                yolo_out   = yolo_model(frame, verbose=False, conf=conf_thresh)
                boxes      = yolo_out[0].boxes
                active_ids = []
                detections = []
                max_risk   = 0

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id   = int(box.cls[0])
                    cls_name = yolo_model.names[cls_id]
                    active_ids.append(i)

                    dist    = depth_estimator.get_distance(
                        depth_raw, (x1, y1, x2, y2)
                    ) if depth_raw is not None else 20.0
                    ttc     = ttc_calc.update(i, dist)
                    ttc_cap = min(ttc, 30) if ttc < 999 else 30
                    obs     = np.array(
                        [dist, ttc_cap, float(cls_id),
                         float(weather_code), 0.8],
                        dtype=np.float32
                    )
                    alert, _ = rl_agent.predict(obs, deterministic=True)
                    ds   = 95 if dist<5 else 75 if dist<10 else \
                           55 if dist<15 else 35 if dist<25 else 10
                    ts   = 95 if ttc_cap<2 else 65 if ttc_cap<4 else \
                           35 if ttc_cap<7 else 10
                    risk     = min(100, round(ds*0.35 + ts*0.65))
                    max_risk = max(max_risk, risk)
                    detections.append({
                        'bbox': (x1, y1, x2, y2), 'class': cls_name,
                        'distance': dist, 'ttc': ttc_cap,
                        'alert': int(alert), 'risk': risk
                    })

                ttc_calc.cleanup(active_ids)
                fps_actual = 1.0 / max(time.time()-t0, 0.001)
                frame, highest_alert = draw_dashboard(
                    frame, detections, conditions, max_risk, fps_actual
                )
                if enable_sound:
                    sound.play(highest_alert)

                stframe.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_container_width=True
                )
                if show_depth and depth_raw is not None:
                    stframe_depth.image(
                        cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB),
                        channels="RGB", use_container_width=True
                    )

                min_ttc = min(
                    (d['ttc'] for d in detections if d['ttc'] < 30),
                    default=999
                )
                ttc_str = f"{min_ttc}s" if min_ttc < 30 else "Safe"
                risk_ph.metric("🔴 Risk Score", f"{max_risk}/100")
                obj_ph.metric("📦 Objects",     len(detections))
                ttc_ph.metric("⏱️ Min TTC",     ttc_str)
                cond_ph.metric("🌤️ Conditions", conditions[0])

                frame_num += 1

            cap.release()
            st.info("Live feed stopped.")

# ══ TAB 2: RESULTS ════════════════════════════════
with tab2:
    st.markdown("### Model Performance Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall mAP50",   "0.777", "+0.35 vs baseline")
    col2.metric("Training Images", "3,438", "10 classes")
    col3.metric("RL Training",     "200K",  "PPO steps")

    st.markdown("---")
    st.markdown("#### Per-Class Detection Accuracy")

    import pandas as pd
    results_df = pd.DataFrame({
        'Class':    ['bike','autorickshaw','car','bus',
                     'bicycle','cycle','motorbike',
                     'person','tractor'],
        'AP50':     [0.995, 0.984, 0.967, 0.950,
                     0.888, 0.862, 0.800, 0.052, 0.492],
        'Grade':    ['Excellent','Excellent','Excellent',
                     'Excellent','Excellent','Excellent',
                     'Good','Limited','Limited']
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### System Comparison")
    comp_df = pd.DataFrame({
        'System':   ['Tesla ADAS','Mobileye','VisionGuard'],
        'Hardware': ['LiDAR+Radar+Camera',
                     'Proprietary Camera+Chip',
                     'Single RGB Camera Only'],
        'Cost':     ['$10,000+','$1,000+','~$0'],
        'Indian Road Classes': ['No','No','Yes']
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ══ TAB 3: ABOUT ══════════════════════════════════
with tab3:
    st.markdown("""
    ### About VisionGuard

    VisionGuard is a camera-only Advanced Driver Assistance
    System built for Indian road conditions as a BTech
    Deep Learning project.

    #### Pipeline Modules
    1. **YOLOv8** — Fine-tuned object detector (mAP50: 0.777)
    2. **DepthAnything v2** — Monocular depth estimation
    3. **TTC Calculator** — Time-to-collision via regression
    4. **Conditions Detector** — Night/Rain/Fog detection
    5. **PPO RL Agent** — Intelligent alert policy

    #### Key Innovation
    Replaces expensive LiDAR sensors (USD 4,000-75,000)
    with a single RGB camera using deep learning — making
    ADAS accessible to every vehicle owner.

    #### Alert Levels
    | Level | Trigger | Action |
    |-------|---------|--------|
    | SAFE | TTC > 10s | Silent |
    | CAUTION | TTC 6-10s | Soft beep |
    | WARNING | TTC 2.5-6s | Urgent beep |
    | BRAKE NOW | TTC < 2.5s | Alarm |

    #### Input Modes
    | Mode | Use Case |
    |------|----------|
    | Upload Video | Testing with dashcam footage |
    | Phone Camera | Live demo using IP Webcam app |
    """)