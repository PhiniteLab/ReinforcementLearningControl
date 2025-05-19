import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# --- Environment using error-based states and dynamic composite reference with delta_u penalty ---
class FirstOrderPlant:
    def __init__(
        self,
        dt: float,
        ref_seq: np.ndarray,
        error_bins: np.ndarray,
        lam: float = 0.01,
        beta: float = 0.1
    ):
        """
        dt: time step
        ref_seq: array of reference signal values over time
        error_bins: boundaries for discretizing the tracking error
        lam: weight on control effort in reward
        beta: weight on change in control input (delta_u) in reward
        """
        self.dt = dt
        self.ref_seq = ref_seq
        self.error_bins = error_bins
        self.lam = lam
        self.beta = beta
        self.actions = np.arange(-10, 11)  # possible control torques
        self.num_actions = len(self.actions)
        self.num_states = len(error_bins) + 1
        self.prev_u = 0.0  # to track previous action

    def discretize_error(self, e: float) -> int:
        """Map continuous error to discrete state via bins."""
        idx = np.digitize(e, self.error_bins)
        return int(np.clip(idx, 0, self.num_states - 1))

    def step(self, x: float, t: int, a: int) -> Tuple[float, float, int, float]:
        """
        Perform one time step.
        x: current output
        t: time index into ref_seq
        a: action index
        Returns: x_next, reward, next_state, applied control u
        """
        u = self.actions[a]
        # penalize change in control input
        delta_u = u - self.prev_u
        self.prev_u = u

        # discrete-time update x_dot = -x + u
        x_next = x + self.dt * (-x + u)

        # reference and error
        r = self.ref_seq[t]
        e = x_next - r

        # reward includes error, control effort, and delta_u penalty
        reward = -(10 * e**2 + self.lam * u**2 + self.beta * delta_u**2)

        s_next = self.discretize_error(e)
        return x_next, reward, s_next, u

    def reset(self) -> Tuple[float, int, int]:
        """Reset plant to initial condition x=0, reset prev_u, and state at t=0."""
        x0 = 0.0
        self.prev_u = 0.0
        e0 = x0 - self.ref_seq[0]
        s0 = self.discretize_error(e0)
        return x0, s0, 0


# --- Q-Learning Agent ---
class QLearningAgent:
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        epsilon: float = 0.2,
        alpha: float = 0.1,
        gamma: float = 0.99
    ):
        """
        num_states: number of discrete states
        num_actions: number of possible actions
        epsilon: exploration rate
        alpha: learning rate
        gamma: discount factor
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((num_states, num_actions))

    def select_action(self, s: int) -> int:
        """ε-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return int(np.argmax(self.Q[s]))

    def update(self, s: int, a: int, r: float, s_next: int) -> None:
        """Q-learning update rule."""
        target = r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay_epsilon(self, decay: float = 0.995, min_epsilon: float = 0.01) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon * decay, min_epsilon)


def generate_composite_reference(
    time: np.ndarray, seed: int = None
) -> np.ndarray:
    """Create a composite reference using sine, cosine, and noise."""
    if seed is not None:
        np.random.seed(seed)
    ref = (
        2.0 * np.sin(0.5 * time) +
        1.5 * np.cos(1.0 * time) +
        0.05 * np.random.randn(len(time))
    )
    return np.clip(ref + 5.0, 0.0, 10.0)


def train_agent(
    env: FirstOrderPlant,
    agent: QLearningAgent,
    episodes: int
) -> List[float]:
    """Train agent over specified episodes; returns episode reward history."""
    T = len(env.ref_seq)
    reward_history = []
    for ep in range(episodes):
        x, s, t = env.reset()
        ep_reward = 0.0
        for _ in range(T):
            a = agent.select_action(s)
            x, r, s_next, _ = env.step(x, t, a)
            agent.update(s, a, r, s_next)
            s, t = s_next, t + 1
            ep_reward += r
        reward_history.append(ep_reward)
        agent.decay_epsilon()
    return reward_history


def simulate_policy(
    env: FirstOrderPlant,
    agent: QLearningAgent
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run greedy policy to produce time, response, and control histories."""
    T = len(env.ref_seq)
    x = 0.0
    x_hist = np.zeros(T)
    u_hist = np.zeros(T)
    for t in range(T):
        e = x - env.ref_seq[t]
        s = env.discretize_error(e)
        a = int(np.argmax(agent.Q[s]))
        u = env.actions[a]
        x = x + env.dt * (-x + u)
        x_hist[t] = x
        u_hist[t] = u
    time = np.arange(T) * env.dt
    return time, x_hist, u_hist


def plot_results(
    time: np.ndarray,
    ref_seq: np.ndarray,
    x_hist: np.ndarray,
    u_hist: np.ndarray,
    reward_history: List[float]
) -> None:
    """Plot training rewards, reference vs. response, and control signal."""
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


# --- Main Execution for dt=0.01 --- #
if __name__ == "__main__":
    dt = 0.01
    total_time = 5.0
    time = np.arange(0, total_time, dt)
    ref_seq = generate_composite_reference(time, seed=42)
    error_bins = np.linspace(-5, 5, 21)

    env = FirstOrderPlant(dt, ref_seq, error_bins, lam=0.001, beta=1.0)
    agent = QLearningAgent(env.num_states, env.num_actions,
                           epsilon=0.5, alpha=0.05, gamma=0.0)

    rewards = train_agent(env, agent, episodes=1000)
    time_out, x_hist, u_hist = simulate_policy(env, agent)
    plot_results(time_out, ref_seq, x_hist, u_hist, rewards)
