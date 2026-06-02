
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class AlertPolicyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.step_count = 0
        self.max_steps  = 300
        self.observation_space = spaces.Box(
            low =np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([80.0,30.0, 9.0, 3.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.correct_alerts = 0
        self.missed_alerts  = 0
        self.false_alerts   = 0

    def _generate_scenario(self):
        if np.random.rand() < 0.4:
            distance = np.random.uniform(2, 15)
            ttc      = np.random.uniform(0.5, 4)
        elif np.random.rand() < 0.3:
            distance = np.random.uniform(15, 30)
            ttc      = np.random.uniform(4, 10)
        else:
            distance = np.random.uniform(30, 80)
            ttc      = np.random.uniform(10, 30)
        object_class = np.random.randint(0, 10)
        weather      = np.random.randint(0, 4)
        time_of_day  = np.random.uniform(0, 1)
        return np.array([distance, ttc, float(object_class),
                         float(weather), time_of_day], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state      = self._generate_scenario()
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        distance = float(self.state[0])
        ttc      = float(self.state[1])
        weather  = float(self.state[3])
        reward   = self._calculate_reward(action, distance, ttc, weather)
        self.state      = self._generate_scenario()
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self.state, reward, done, False, {}

    def _calculate_reward(self, action, distance, ttc, weather):
        weather_factor = 1.4 if weather in [2, 3] else 1.0
        critical_ttc  = 2.5  * weather_factor
        warning_ttc   = 6.0  * weather_factor
        critical_dist = 8.0  * weather_factor
        warning_dist  = 20.0 * weather_factor
        is_critical = (ttc < critical_ttc) or (distance < critical_dist)
        is_warning  = (ttc < warning_ttc)  or (distance < warning_dist)
        is_safe     = not is_critical and not is_warning
        if is_critical:
            if action == 3:   return +20
            elif action == 2: return +8
            elif action == 1: return -10
            else:             return -25
        elif is_warning:
            if action == 2:   return +15
            elif action == 1: return +8
            elif action == 3: return -5
            else:             return -12
        else:
            if action == 0:   return +5
            elif action == 1: return -3
            elif action == 2: return -8
            else:             return -15
