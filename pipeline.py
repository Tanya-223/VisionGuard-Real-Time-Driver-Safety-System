# pipeline.py
# Core pipeline — all processing logic here

import cv2
import numpy as np
import threading
import time
from PIL import Image
from collections import deque

# ── Sound Alert System ─────────────────────────────
class SoundAlert:
    def __init__(self):
        self.last_alert_time  = 0
        self.min_gap_seconds  = 2.5  # don't beep more than once per 2.5s
        self.current_level    = 0
        self._playing         = False

    def play(self, level):
        """
        level 0 = silent
        level 1 = mild beep
        level 2 = warning beep
        level 3 = critical alarm
        """
        now = time.time()
        # Only play if enough time passed and level increased
        if level == 0:
            return
        if level <= self.current_level and \
           now - self.last_alert_time < self.min_gap_seconds:
            return
        if self._playing:
            return

        self.current_level   = level
        self.last_alert_time = now

        sound_files = {
            1: 'sounds/alert_mild.wav',
            2: 'sounds/alert_warning.wav',
            3: 'sounds/alert_critical.wav'
        }
        sound_file = sound_files.get(level)
        if sound_file:
            threading.Thread(
                target=self._play_sound,
                args=(sound_file,),
                daemon=True
            ).start()

    def _play_sound(self, filepath):
        self._playing = True
        try:
            import winsound  # Windows
            winsound.PlaySound(filepath, winsound.SND_FILENAME)
        except ImportError:
            try:
                import subprocess  # Mac/Linux
                subprocess.Popen(
                    ['aplay', filepath],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception:
                pass  # Sound failed silently — pipeline still works
        self._playing = False


# ── Depth Estimator ────────────────────────────────
class DepthEstimator:
    def __init__(self):
        from transformers import pipeline as hf_pipeline
        print("Loading depth model...")
        self.pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf"
        )
        print("Depth model ready!")

    def get_depth_map(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb)
        result    = self.pipe(pil_img)
        depth_raw = np.array(result["depth"], dtype=np.float32)
        h, w      = frame_bgr.shape[:2]
        depth_raw = cv2.resize(depth_raw, (w, h))
        depth_norm = cv2.normalize(depth_raw, None, 0, 255,
                                   cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        return depth_raw, depth_colored

    def get_distance(self, depth_raw, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        cx1 = x1 + (x2-x1)//4
        cy1 = y1 + (y2-y1)//4
        cx2 = x1 + 3*(x2-x1)//4
        cy2 = y1 + 3*(y2-y1)//4
        h, w = depth_raw.shape
        cx1,cx2 = max(0,min(cx1,w-1)), max(0,min(cx2,w-1))
        cy1,cy2 = max(0,min(cy1,h-1)), max(0,min(cy2,h-1))
        region  = depth_raw[cy1:cy2, cx1:cx2]
        if region.size == 0:
            return 99.0
        median_val  = float(np.median(region))
        max_d       = float(depth_raw.max())
        min_d       = float(depth_raw.min())
        normalized  = (median_val - min_d) / (max_d - min_d + 1e-6)
        return round(2.0 + ((1.0 - normalized) * 48.0), 1)


# ── TTC Calculator ─────────────────────────────────
class TTCCalculator:
    def __init__(self, fps=30):
        self.fps     = fps
        self.history = {}

    def update(self, track_id, distance):
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=8)
        self.history[track_id].append(distance)
        hist = list(self.history[track_id])
        if len(hist) < 3:
            return 999.0
        x     = np.arange(len(hist))
        slope = np.polyfit(x, hist, 1)[0]
        vel   = -slope * self.fps
        if vel <= 0.2:
            return 999.0
        return round(max(0.0, distance / vel), 1)

    def cleanup(self, active_ids):
        for tid in list(self.history.keys()):
            if tid not in active_ids:
                del self.history[tid]


# ── Conditions Detector ────────────────────────────
def detect_conditions(frame_bgr):
    gray       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    blur       = cv2.Laplacian(gray, cv2.CV_64F).var()
    glare      = np.sum(gray > 245) / gray.size
    conditions = []
    if brightness < 45:   conditions.append('NIGHT')
    elif brightness < 75: conditions.append('DIM')
    if blur < 80:         conditions.append('FOG')
    elif blur < 150:      conditions.append('RAIN')
    if glare > 0.12:      conditions.append('GLARE')
    if not conditions:    conditions.append('CLEAR')
    return conditions


# ── Dashboard Overlay ──────────────────────────────
def draw_dashboard(frame, detections, conditions,
                   max_risk, fps_actual):
    """
    Draws full overlay including:
    - Dark top bar
    - Conditions badge
    - Risk score bar
    - Mini dashboard panel (bottom right)
    - Bounding boxes with labels
    - Big alert text
    """
    h, w = frame.shape[:2]

    # ── Colours per alert level ──
    ALERT_COLORS = {
        0: (0, 220, 100),   # Green
        1: (0, 230, 230),   # Yellow
        2: (0, 140, 255),   # Orange
        3: (0, 0, 255),     # Red
    }
    ALERT_TEXTS = {
        0: '',
        1: 'CAUTION',
        2: 'WARNING!',
        3: 'BRAKE NOW!'
    }

    # ── Dark top bar ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (12, 12, 20), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # ── Condition badge ──
    COND_INFO = {
        'NIGHT': ('NIGHT MODE',    (255, 200, 50)),
        'FOG':   ('FOG DETECTED',  (210, 210, 210)),
        'RAIN':  ('RAIN DETECTED', (180, 180, 255)),
        'GLARE': ('GLARE WARNING', (0,   220, 255)),
        'DIM':   ('LOW LIGHT',     (180, 180, 100)),
        'CLEAR': ('CLEAR',         (100, 255, 100)),
    }
    cond_key  = next((c for c in
                      ['NIGHT','FOG','RAIN','GLARE','DIM']
                      if c in conditions), 'CLEAR')
    cond_text, cond_color = COND_INFO[cond_key]
    cv2.putText(frame, cond_text, (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, cond_color, 2)

    # ── Risk bar ──
    risk_color = (0,220,100) if max_risk < 30 else \
                 (0,210,210) if max_risk < 55 else \
                 (0,140,255) if max_risk < 75 else \
                 (0,0,255)
    bar_w = int((w - 30) * max_risk / 100)
    cv2.rectangle(frame, (14, 48), (w-14, 70), (45,45,45), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (14, 48), (14+bar_w, 70), risk_color, -1)
    cv2.putText(frame, f'RISK: {max_risk}/100', (18, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)

    # ── FPS top right ──
    cv2.putText(frame, f'FPS:{fps_actual:.1f}',
                (w-90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,160,160), 1)

    # ── Mini dashboard panel (bottom right) ──
    pw, ph = 160, 130
    px, py = w-pw-10, h-ph-10
    panel  = frame.copy()
    cv2.rectangle(panel, (px, py), (px+pw, py+ph), (15,15,25), -1)
    cv2.addWeighted(panel, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (px, py), (px+pw, py+ph),
                  (60,60,80), 1)

    # Panel title
    cv2.putText(frame, "DASHBOARD", (px+10, py+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,180), 1)

    # Objects count
    obj_count = len(detections)
    cv2.putText(frame, f"Objects: {obj_count}", (px+10, py+40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)

    # Closest object
    if detections:
        closest = min(detections, key=lambda x: x['distance'])
        cv2.putText(frame,
                    f"Closest:{closest['distance']}m",
                    (px+10, py+60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200,200,200), 1)
        # TTC
        ttc_val = closest['ttc']
        ttc_str = f"TTC: {ttc_val}s" if ttc_val < 30 else "TTC: Safe"
        ttc_col = (0,0,255) if ttc_val < 3 else \
                  (0,165,255) if ttc_val < 6 else \
                  (100,255,100)
        cv2.putText(frame, ttc_str, (px+10, py+80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, ttc_col, 1)

    # Risk in panel
    cv2.putText(frame, f"Risk: {max_risk}/100",
                (px+10, py+100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48, risk_color, 1)

    # Condition in panel
    cv2.putText(frame, cond_text[:10],
                (px+10, py+118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42, cond_color, 1)

    # ── Bounding boxes ──
    highest_alert = 0
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_name        = det['class']
        distance        = det['distance']
        ttc             = det['ttc']
        alert_id        = det['alert']
        box_color       = ALERT_COLORS[alert_id]

        if alert_id > highest_alert:
            highest_alert = alert_id

        thickness = max(1, alert_id + 1)
        cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, thickness)

        ttc_str = f" T:{ttc}s" if ttc < 30 else ""
        label   = f"{cls_name} {distance}m{ttc_str}"
        lw      = len(label) * 8

        if y1 > 100:
            cv2.rectangle(frame, (x1, y1-26),
                          (x1+lw, y1), box_color, -1)
            cv2.putText(frame, label, (x1+3, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0,0,0), 1)
        else:
            cv2.rectangle(frame, (x1, y2),
                          (x1+lw, y2+26), box_color, -1)
            cv2.putText(frame, label, (x1+3, y2+17),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0,0,0), 1)

    # ── Big alert text bottom center ──
    if highest_alert > 0:
        alert_text  = ALERT_TEXTS[highest_alert]
        alert_color = ALERT_COLORS[highest_alert]
        ts = cv2.getTextSize(alert_text,
                             cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0]
        tx = (w - ts[0]) // 2
        cv2.rectangle(frame,
                      (tx-12, h-ph-60),
                      (tx+ts[0]+12, h-ph-18),
                      (0,0,0), -1)
        cv2.putText(frame, alert_text,
                    (tx, h-ph-24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, alert_color, 3)

    return frame, highest_alert