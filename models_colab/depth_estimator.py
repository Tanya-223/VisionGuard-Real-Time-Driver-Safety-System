
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline as hf_pipeline

class DepthEstimator:
    def __init__(self):
        print("Loading DepthAnything v2...")
        self.pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf"
        )
        print("Depth model loaded!")

    def get_depth_map(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        result = self.pipe(pil_image)
        depth_raw = np.array(result["depth"], dtype=np.float32)
        h, w = frame_bgr.shape[:2]
        depth_raw = cv2.resize(depth_raw, (w, h))
        depth_norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        return depth_raw, depth_colored

    def get_object_distance(self, depth_raw, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        cx1 = x1 + (x2 - x1) // 4
        cy1 = y1 + (y2 - y1) // 4
        cx2 = x1 + 3 * (x2 - x1) // 4
        cy2 = y1 + 3 * (y2 - y1) // 4
        h, w = depth_raw.shape
        cx1,cx2 = max(0,min(cx1,w-1)), max(0,min(cx2,w-1))
        cy1,cy2 = max(0,min(cy1,h-1)), max(0,min(cy2,h-1))
        region = depth_raw[cy1:cy2, cx1:cx2]
        if region.size == 0:
            return 99.0
        median_val  = float(np.median(region))
        max_depth   = float(depth_raw.max())
        min_depth   = float(depth_raw.min())
        depth_range = max_depth - min_depth + 1e-6
        normalized  = (median_val - min_depth) / depth_range
        distance_m  = 2.0 + ((1.0 - normalized) * 48.0)
        return round(distance_m, 1)
