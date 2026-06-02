
import cv2
import numpy as np

class ConditionsDetector:
    def detect(self, frame_bgr):
        conditions = []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 45:       conditions.append("NIGHT")
        elif brightness < 75:     conditions.append("DIM")
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 80:       conditions.append("FOG")
        elif blur_score < 150:    conditions.append("RAIN")
        very_bright = np.sum(gray > 245)
        if very_bright / gray.size > 0.12: conditions.append("GLARE")
        if not conditions:        conditions.append("CLEAR")
        return conditions

    def get_display_info(self, conditions):
        if "NIGHT" in conditions:  return "NIGHT MODE", (255, 200, 50)
        elif "FOG" in conditions:  return "FOG DETECTED", (200, 200, 200)
        elif "RAIN" in conditions: return "RAIN DETECTED", (200, 200, 255)
        elif "GLARE" in conditions:return "GLARE WARNING", (0, 220, 255)
        elif "DIM" in conditions:  return "LOW LIGHT", (180, 180, 100)
        else:                      return "CONDITIONS: CLEAR", (100, 255, 100)
