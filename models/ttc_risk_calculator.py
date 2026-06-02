
import numpy as np
from collections import deque

class TTCCalculator:
    def __init__(self, fps=30, history_size=8):
        self.fps = fps
        self.history_size = history_size
        self.distance_history = {}

    def update(self, track_id, current_distance):
        if track_id not in self.distance_history:
            self.distance_history[track_id] = deque(maxlen=self.history_size)
        self.distance_history[track_id].append(current_distance)
        history = list(self.distance_history[track_id])
        if len(history) < 3:
            return 999.0
        x = np.arange(len(history))
        slope = np.polyfit(x, history, 1)[0]
        velocity_per_second = -slope * self.fps
        if velocity_per_second <= 0.2:
            return 999.0
        ttc = current_distance / velocity_per_second
        return round(max(0.0, ttc), 1)

    def remove_lost_tracks(self, active_ids):
        lost = [tid for tid in self.distance_history if tid not in active_ids]
        for tid in lost:
            del self.distance_history[tid]


class RiskCalculator:
    CLASS_RISK = {
        "person":1.6,"bicycle":1.4,"cycle":1.4,"bike":1.3,
        "motorbike":1.3,"autorickshaw":1.2,"tractor":1.2,
        "bus":1.1,"truck":1.1,"car":1.0
    }

    def calculate_risk(self, distance, ttc, object_class, conditions=[]):
        if distance < 5:      dist_score = 95
        elif distance < 10:   dist_score = 75
        elif distance < 15:   dist_score = 55
        elif distance < 25:   dist_score = 35
        elif distance < 35:   dist_score = 20
        else:                 dist_score = 8
        if ttc < 1.5:         ttc_score = 98
        elif ttc < 2.5:       ttc_score = 80
        elif ttc < 4:         ttc_score = 60
        elif ttc < 7:         ttc_score = 35
        elif ttc < 12:        ttc_score = 15
        else:                 ttc_score = 5
        base_risk = (dist_score * 0.35) + (ttc_score * 0.65)
        base_risk *= self.CLASS_RISK.get(object_class, 1.0)
        if "NIGHT" in conditions: base_risk *= 1.3
        if "FOG"   in conditions: base_risk *= 1.25
        if "RAIN"  in conditions: base_risk *= 1.2
        return min(100, round(base_risk))

    def get_alert_info(self, risk_score):
        if risk_score >= 75:   return "BRAKE NOW!",  (0, 0, 255)
        elif risk_score >= 55: return "WARNING!",    (0, 100, 255)
        elif risk_score >= 30: return "CAUTION",     (0, 255, 255)
        else:                  return "",             (0, 255, 0)
