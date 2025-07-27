import numpy as np
import gymnasium as gym
from gymnasium import spaces


class UAVSecureCommEnv(gym.Env):
    def __init__(self):
        super(UAVSecureCommEnv, self).__init__()

        self.max_steps = 30
        self.step_count = 0

        self.scene_bounds = {
            'x': (-100, 100),
            'y': (-100, 100),
            'step': (0, 30)
        }

        self.bs_pos = np.array([0, 0, 0])
        self.eve_pos = np.array([100.0, -100.0, 100.0])
        self.jam_pos = np.array([-100.0, -100.0, 0.0])
        self.uav_pos = np.array([-100.0, 0.0, 100.0])
        self.uav_goal = np.array([100.0, 100.0, 100.0])


        self.n_antennas = 4
        self.P_b_max = 120.0
        self.P_j = 0.5
        self.noise_power = 10 ** (-90 / 10) / 1000  # W


        self.dx_max = 20.0
        self.dy_max = 20.0
        self.max_step_xy = 20.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2 + 2 * self.n_antennas,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        self.total_secrecy_rate = 0.0
        self.total_reward = 0.0

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.uav_pos = np.array([-100.0, 0.0, 100.0])
        self.total_secrecy_rate = 0.0
        self.total_reward = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.uav_pos[0] / 100.0,
            self.uav_pos[1] / 100.0,
            self.bs_pos[0] / 100.0,
            self.bs_pos[1] / 100.0,
            self.jam_pos[0] / 100.0,
            self.jam_pos[1] / 100.0,
            self.eve_pos[0] / 100.0,
            self.eve_pos[1] / 100.0,
            self.uav_goal[0] / 100.0,
            self.uav_goal[1] / 100.0,
            self.step_count / self.max_steps,
        ])

    def _free_space_loss(self, d):
        return 1.0 / (d ** 2 + 1e-6)

    def _calculate_azimuth(self, tx_pos, rx_pos):
        delta = tx_pos - rx_pos
        return np.arctan2(delta[1], delta[0])

    def _array_response(self, azimuth_rad):
        N = self.n_antennas
        d = 0.5
        k = 2 * np.pi
        n = np.arange(N)
        return np.exp(-1j * k * d * n * np.sin(azimuth_rad)) / np.sqrt(N)

    def _beamforming_gain(self, h, w):
        return np.abs(np.dot(np.conj(h), w)) ** 2

    def step(self, action):
        dx, dy = action[0:2]
        beam_real = action[2:2 + self.n_antennas]
        beam_imag = action[2 + self.n_antennas:]
        w = beam_real + 1j * beam_imag
        norm = np.linalg.norm(w)
        if norm < 1e-3:
            w = np.ones_like(w) / np.sqrt(len(w))
        else:
            w = w / norm

        move_vec = np.array([dx * self.dx_max, dy * self.dy_max, 0.0])

        delta_x = abs(self.uav_pos[0] - self.uav_goal[0])
        delta_y = abs(self.uav_pos[1] - self.uav_goal[1])
        can_reach = (delta_x <= self.max_step_xy) and (delta_y <= self.max_step_xy)

        if self.step_count == self.max_steps - 1 and can_reach:
            self.uav_pos[0] = self.uav_goal[0]
            self.uav_pos[1] = self.uav_goal[1]
        else:
            self.uav_pos += move_vec
        self.uav_pos[0] = np.clip(self.uav_pos[0], *self.scene_bounds['x'])
        self.uav_pos[1] = np.clip(self.uav_pos[1], *self.scene_bounds['y'])

        d_bu = np.linalg.norm(self.bs_pos - self.uav_pos)
        d_be = np.linalg.norm(self.bs_pos - self.eve_pos)
        d_ju = np.linalg.norm(self.jam_pos - self.uav_pos)
        d_je = np.linalg.norm(self.jam_pos - self.eve_pos)

        az_bu = self._calculate_azimuth(self.uav_pos, self.bs_pos)
        h_bu = self._array_response(az_bu) * np.sqrt(self._free_space_loss(d_bu))
        az_be = self._calculate_azimuth(self.eve_pos, self.bs_pos)
        h_be = self._array_response(az_be) * np.sqrt(self._free_space_loss(d_be))
        h_ju = self._free_space_loss(d_ju)
        h_je = self._free_space_loss(d_je)

        gain_bu = self._beamforming_gain(h_bu, w)
        gain_be = self._beamforming_gain(h_be, w)

        sinr_uav = (self.P_b_max * gain_bu) / (self.P_j * h_ju + self.noise_power)
        sinr_eve = (self.P_b_max * gain_be) / (self.P_j * h_je + self.noise_power)
        secrecy_rate = np.log2(1 + sinr_uav) - np.log2(1 + sinr_eve)

        reward = secrecy_rate
        self.total_reward += reward
        self.total_secrecy_rate += max(np.log2(1 + sinr_uav) - np.log2(1 + sinr_eve), 0)
        self.step_count += 1
        done = self.step_count >= self.max_steps
        info = {
            "secrecy_rate": max(np.log2(1 + sinr_uav) - np.log2(1 + sinr_eve), 0),
            "sinr_uav": sinr_uav,
            "sinr_eve": sinr_eve,
        }
        if done:
            if can_reach:
                reward += 100.0
                info["goal_reward"] = 100
            else:
                reward -= 100.0
                info["goal_reward"] = -100
        return self._get_obs(), reward, False, done, info

    def render(self, mode='human'):
        print(f"Step: {self.step_count}, UAV Pos: {self.uav_pos}")


def test_env():
    env = UAVSecureCommEnv()
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, _, done, info = env.step(action)
        print(obs)
        total_reward += reward

        print(f"Step: {env.step_count}")
        print(f"UAV Position: {obs[:3]}")
        print(f"Reward: {reward:.4f}")
        print(f"Secrecy Rate: {info['secrecy_rate']:.4f}")
        print(f"SINR UAV: {info['sinr_uav']:.4e}")
        print(f"SINR Eve: {info['sinr_eve']:.4e}")
        print("-" * 40)

    print(f"Total reward over episode: {total_reward:.4f}")


def test_normalization():
    env = UAVSecureCommEnv()

    # 测试观测值范围
    for _ in range(100):
        obs = env.reset()[0]
        assert all(-1 <= x <= 1 for x in obs), f"Out-of-Bound: {obs}"

        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs), "Mismatched Space"


if __name__ == "__main__":
    test_env()
