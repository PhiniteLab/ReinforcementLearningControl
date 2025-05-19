import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# --- 1) Environment (unchanged) ---
class FirstOrderPlant:
    def __init__(
        self,
        dt: float,
        ref_seq: np.ndarray,
        error_bins: np.ndarray,
        error_weight: float = 10.0,
        lam: float = 0.01,
        beta: float = 0.1
    ):
        self.dt = dt
        self.ref_seq = ref_seq
        self.error_bins = error_bins
        self.error_weight = error_weight
        self.lam = lam
        self.beta = beta
        self.actions = np.arange(-10, 11, 0.5)
        self.num_actions = len(self.actions)
        self.num_states = len(error_bins) + 1
        self.prev_u = 0.0

    def discretize_error(self, e: float) -> int:
        idx = np.digitize(e, self.error_bins)
        return int(np.clip(idx, 0, self.num_states - 1))

    def step(self, x: float, t: int, a: int) -> Tuple[float, float, int, float]:
        u = self.actions[a]
        delta_u = u - self.prev_u
        self.prev_u = u

        x_next = x + self.dt * (-x + u)
        r = self.ref_seq[t]
        e = x_next - r

        reward = -(
            self.error_weight * e**2 +
            self.lam * u**2 +
            self.beta * delta_u**2
        )
        s_next = self.discretize_error(e)
        return x_next, reward, s_next, u

    def reset(self) -> Tuple[float, int, int]:
        x0 = 0.0
        self.prev_u = 0.0
        e0 = x0 - self.ref_seq[0]
        s0 = self.discretize_error(e0)
        return x0, s0, 0

# --- 2) SARSA Agent (On-Policy TD Control) ---
class SarsaAgent:
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        gamma: float = 0.99,
        alpha: float = 0.1,
        epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        decay: float = 0.995
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

        # Q-table
        self.Q = np.zeros((num_states, num_actions))

    def select_action(self, s: int) -> int:
        """ε-greedy action selection on Q."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.Q[s]))

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int):
        """SARSA update rule."""
        td_target = r + self.gamma * self.Q[s_next, a_next]
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

        # decay epsilon
        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)

# --- 3) Reference generator (unchanged) ---
def generate_composite_reference(
    time: np.ndarray, seed: int = None
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    ref = (
        2.0 * np.sin(0.5 * time) +
        1.5 * np.cos(1.0 * time) +
        0.05 * np.random.randn(len(time))
    )
    return np.clip(ref + 5.0, 0.0, 10.0)

# --- 4) Training loop using SARSA ---
def train_agent(
    env: FirstOrderPlant,
    agent: SarsaAgent,
    episodes: int
) -> List[float]:
    T = len(env.ref_seq)
    reward_history = []
    for ep in range(episodes):
        x, s, t = env.reset()
        a = agent.select_action(s)
        ep_reward = 0.0

        for _ in range(T):
            x2, r, s2, _ = env.step(x, t, a)
            a2 = agent.select_action(s2)
            agent.update(s, a, r, s2, a2)

            x, s, a = x2, s2, a2
            t += 1
            ep_reward += r

        reward_history.append(ep_reward)
    return reward_history

# --- 5) Simulation of final policy ---
def simulate_policy(
    env: FirstOrderPlant,
    agent: SarsaAgent
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = len(env.ref_seq)
    x = 0.0
    x_hist = np.zeros(T)
    u_hist = np.zeros(T)
    for t in range(T):
        e = x - env.ref_seq[t]
        s = env.discretize_error(e)
        a = agent.select_action(s)  # now purely greedy as epsilon has decayed
        u = env.actions[a]
        x = x + env.dt * (-x + u)
        x_hist[t] = x
        u_hist[t] = u
    time = np.arange(T) * env.dt
    return time, x_hist, u_hist

# --- 6) Plot results (unchanged) ---
def plot_results(
    time: np.ndarray,
    ref_seq: np.ndarray,
    x_hist: np.ndarray,
    u_hist: np.ndarray,
    reward_history: List[float]
) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    axs[0].plot(reward_history)
    axs[0].set(title="Episode Rewards", xlabel="Episode", ylabel="Total Reward")
    axs[0].grid(True)

    axs[1].plot(time, ref_seq, '--', label="Reference")
    axs[1].plot(time, x_hist, label="Response")
    axs[1].set(title="Reference vs Response", xlabel="Time (s)", ylabel="x(t)")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].step(time, u_hist, where="post")
    axs[2].set(title="Control Signal", xlabel="Time (s)", ylabel="u(t)")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

# --- 7) Main Execution ---
if __name__ == "__main__":
    dt = 0.01
    total_time = 15.0
    time = np.arange(0, total_time, dt)
    ref_seq = generate_composite_reference(time, seed=42)
    error_bins = np.linspace(-5, 5, 21)

    env = FirstOrderPlant(
        dt, ref_seq, error_bins,
        error_weight=500.0,
        lam=0.01, beta=20.0
    )
    agent = SarsaAgent(
        env.num_states, env.num_actions,
        gamma=0.99, alpha=0.1,
        epsilon=1.0, min_epsilon=0.01, decay=0.995
    )

    rewards = train_agent(env, agent, episodes=1000)
    time_out, x_hist, u_hist = simulate_policy(env, agent)
    plot_results(time_out, ref_seq, x_hist, u_hist, rewards)
