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
        error_weight: float = 10.0,
        lam: float = 0.01,
        beta: float = 0.1
    ):
        """
        dt: time step
        ref_seq: array of reference signal values over time
        error_bins: boundaries for discretizing the tracking error
        error_weight: coefficient on e^2 in the reward
        lam: weight on control effort in reward
        beta: weight on change in control input (delta_u) in reward
        """
        self.dt = dt
        self.ref_seq = ref_seq
        self.error_bins = error_bins
        self.error_weight = error_weight
        self.lam = lam
        self.beta = beta
        self.actions = np.arange(-10, 11, 0.5)  # possible control torques
        self.num_actions = len(self.actions)
        self.num_states = len(error_bins) + 1
        self.prev_u = 0.0  # to track previous action

    def discretize_error(self, e: float) -> int:
        """Map continuous error to discrete state via bins."""
        idx = np.digitize(e, self.error_bins)
        return int(np.clip(idx, 0, self.num_states - 1))

    def step(self, x: float, t: int, a: int) -> Tuple[float, float, int, float]:
        u = self.actions[a]
        delta_u = u - self.prev_u
        self.prev_u = u

        # discrete-time update x_dot = -x + u
        x_next = x + self.dt * (-x + u)
        r = self.ref_seq[t]
        e = x_next - r

        # reward includes error, control effort, and delta_u penalty
        reward = -(
            self.error_weight * e**2 +
            self.lam * u**2 +
            self.beta * delta_u**2
        )

        s_next = self.discretize_error(e)
        return x_next, reward, s_next, u

    def reset(self) -> Tuple[float, int, int]:
        """Reset plant to initial condition x=0, reset prev_u, and state at t=0."""
        x0 = 0.0
        self.prev_u = 0.0
        e0 = x0 - self.ref_seq[0]
        s0 = self.discretize_error(e0)
        return x0, s0, 0


# --- Off-Policy Monte Carlo Control with Weighted Importance Sampling (no normalization) & ε-decay ---
class OffPolicyMC:
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        min_epsilon: float = 0.01,
        decay: float = 0.995
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma

        # behavior policy parameters
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

        # Q(s,a) table and cumulative raw weights C(s,a)
        self.Q = np.zeros((num_states, num_actions))
        self.C = np.zeros((num_states, num_actions))

        # target policy: greedy w.r.t Q
        self.policy = np.zeros(num_states, dtype=int)

    def behavior_prob(self, s: int, a: int) -> float:
        """Probability of picking action a under ε-greedy behavior policy."""
        greedy_a = self.policy[s]
        if a == greedy_a:
            return 1 - self.epsilon + self.epsilon / self.num_actions
        return self.epsilon / self.num_actions

    def select_action(self, s: int) -> int:
        """ε-greedy behavior policy for exploration."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(self.policy[s])

    def update_episode(self, episode: List[Tuple[int,int,float]]):
        """
        Off-policy MC update using Weighted Importance Sampling without normalization.
        episode: list of (state, action, reward) tuples
        """
        G = 0.0
        W = 1.0

        # iterate from end to start
        for s, a, r in reversed(episode):
            G = self.gamma * G + r
            mu = self.behavior_prob(s, a)
            W *= 1.0 / mu

            # accumulate raw weight
            self.C[s, a] += W

            # ordinary weighted update
            self.Q[s, a] += (W / self.C[s, a]) * (G - self.Q[s, a])

            # update target policy greedily
            self.policy[s] = int(np.argmax(self.Q[s]))

            # if behavior deviates from target, stop backing up
            if a != self.policy[s]:
                break

        # decay exploration rate
        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)


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


def train_agent(
    env: FirstOrderPlant,
    agent: OffPolicyMC,
    episodes: int
) -> List[float]:
    """Train agent over specified episodes; returns episode reward history."""
    T = len(env.ref_seq)
    reward_history = []
    for ep in range(episodes):
        x, s, t = env.reset()
        episode = []
        ep_reward = 0.0
        for _ in range(T):
            a = agent.select_action(s)
            x, r, s, _ = env.step(x, t, a)
            episode.append((s, a, r))
            ep_reward += r
            t += 1
        agent.update_episode(episode)
        reward_history.append(ep_reward)
    return reward_history


def simulate_policy(
    env: FirstOrderPlant,
    agent: OffPolicyMC
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run greedy target policy to produce time, response, and control histories."""
    T = len(env.ref_seq)
    x = 0.0
    x_hist = np.zeros(T)
    u_hist = np.zeros(T)
    for t in range(T):
        e = x - env.ref_seq[t]
        s = env.discretize_error(e)
        a = agent.policy[s]
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


# --- Main Execution for dt=0.01 ---
if __name__ == "__main__":
    dt = 0.01
    total_time = 10.0
    time = np.arange(0, total_time, dt)
    ref_seq = generate_composite_reference(time, seed=42)
    error_bins = np.linspace(-5, 5, 21)

    env = FirstOrderPlant(dt, ref_seq, error_bins,
                          error_weight=100.0, lam=0.1, beta=10.0)
    agent = OffPolicyMC(env.num_states, env.num_actions,
                        gamma=0.999, epsilon=0.11,
                        min_epsilon=0.1, decay=0.995)

    rewards = train_agent(env, agent, episodes=1000)
    time_out, x_hist, u_hist = simulate_policy(env, agent)
    plot_results(time_out, ref_seq, x_hist, u_hist, rewards)