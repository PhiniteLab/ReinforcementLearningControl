import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# --- 1) Environment (FirstOrderPlant) ---
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

# --- 2) Agent classes ---
class SarsaAgent:
    """On-policy SARSA(0)."""
    def __init__(self, S, A, gamma=0.99, alpha=0.1,
                 epsilon=1.0, min_eps=0.01, decay=0.995):
        self.S, self.A = S, A
        self.gamma, self.alpha = gamma, alpha
        self.epsilon, self.min_eps, self.decay = epsilon, min_eps, decay
        self.Q = np.zeros((S, A))

    def select(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.A)
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, s2, a2):
        target = r + self.gamma * self.Q[s2, a2]
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        self.epsilon = max(self.epsilon * self.decay, self.min_eps)

class QLearningAgent:
    """Off-policy Q-learning."""
    def __init__(self, S, A, gamma=0.99, alpha=0.1,
                 epsilon=1.0, min_eps=0.01, decay=0.995):
        self.S, self.A = S, A
        self.gamma, self.alpha = gamma, alpha
        self.epsilon, self.min_eps, self.decay = epsilon, min_eps, decay
        self.Q = np.zeros((S, A))

    def select(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.A)
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, s2, _):
        target = r + self.gamma * np.max(self.Q[s2])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        self.epsilon = max(self.epsilon * self.decay, self.min_eps)

class ExpectedSarsaAgent:
    """On-policy Expected SARSA."""
    def __init__(self, S, A, gamma=0.99, alpha=0.1,
                 epsilon=1.0, min_eps=0.01, decay=0.995):
        self.S, self.A = S, A
        self.gamma, self.alpha = gamma, alpha
        self.epsilon, self.min_eps, self.decay = epsilon, min_eps, decay
        self.Q = np.zeros((S, A))

    def select(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.A)
        return int(np.argmax(self.Q[s]))

    def expected_Q(self, s2):
        greedy = np.argmax(self.Q[s2])
        e = self.epsilon
        exp = 0.0
        for a in range(self.A):
            pi = ((1-e) + e/self.A) if a == greedy else (e/self.A)
            exp += pi * self.Q[s2, a]
        return exp

    def update(self, s, a, r, s2, _):
        target = r + self.gamma * self.expected_Q(s2)
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        self.epsilon = max(self.epsilon * self.decay, self.min_eps)

class DoubleQLearningAgent:
    """Off-policy Double Q-learning."""
    def __init__(self, S, A, gamma=0.99, alpha=0.1,
                 epsilon=1.0, min_eps=0.01, decay=0.995):
        self.S, self.A = S, A
        self.gamma, self.alpha = gamma, alpha
        self.epsilon, self.min_eps, self.decay = epsilon, min_eps, decay
        self.Q1 = np.zeros((S, A))
        self.Q2 = np.zeros((S, A))

    def select(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.A)
        return int(np.argmax(self.Q1[s] + self.Q2[s]))

    def update(self, s, a, r, s2, _):
        if np.random.rand() < 0.5:
            a_max = np.argmax(self.Q1[s2])
            target = r + self.gamma * self.Q2[s2, a_max]
            self.Q1[s, a] += self.alpha * (target - self.Q1[s, a])
        else:
            a_max = np.argmax(self.Q2[s2])
            target = r + self.gamma * self.Q1[s2, a_max]
            self.Q2[s, a] += self.alpha * (target - self.Q2[s, a])
        self.epsilon = max(self.epsilon * self.decay, self.min_eps)

    @property
    def Q(self):
        return self.Q1 + self.Q2

# --- 3) Reference generator ---
def generate_composite_reference(time: np.ndarray, seed: int = None) -> np.ndarray:
    if seed is not None: np.random.seed(seed)
    ref = 2*np.sin(0.5*time) + 1.5*np.cos(time) + 0.05*np.random.randn(len(time))
    return np.clip(ref+5, 0, 10)

# --- 4) Run experiment ---
def run_experiment(agent, env, episodes=500):
    T = len(env.ref_seq)
    rewards = []
    for ep in range(episodes):
        x, s, t = env.reset()
        a = agent.select(s)
        total_r = 0.0
        for _ in range(T):
            x2, r, s2, _ = env.step(x, t, a)
            a2 = agent.select(s2)
            agent.update(s, a, r, s2, a2)
            x, s, a, t = x2, s2, a2, t+1
            total_r += r
        rewards.append(total_r)

    # simulate final policy
    x = 0.0
    x_hist = np.zeros(T)
    u_hist = np.zeros(T)
    env.reset()
    for t in range(T):
        s = env.discretize_error(x - env.ref_seq[t])
        a = int(np.argmax(agent.Q[s]))
        x, _, _, u = env.step(x, t, a)
        x_hist[t] = x
        u_hist[t] = u
    return np.array(rewards), x_hist, u_hist

# --- 5) Evaluation metrics ---
def evaluate(x_hist, u_hist, ref_seq, returns, last_N=50):
    mse = np.mean((x_hist - ref_seq)**2)
    tol = 0.1
    out_of_tol = 100 * np.mean(np.abs(x_hist - ref_seq) > tol)
    avg_u = np.mean(np.abs(u_hist))
    avg_ret = np.mean(returns[-last_N:])
    return avg_ret, mse, out_of_tol, avg_u

# --- 6) Main & Plotting ---
if __name__ == "__main__":
    # Setup
    dt = 0.01
    total_time = 10.0
    time = np.arange(0, total_time, dt)
    ref_seq = generate_composite_reference(time, seed=42)
    error_bins = np.linspace(-5, 5, 21)
    env = FirstOrderPlant(dt, ref_seq, error_bins,
                          error_weight=100.0, lam=0.01, beta=10.0)

    agents = {
        'SARSA': SarsaAgent(env.num_states, env.num_actions),
        'Q-Learning': QLearningAgent(env.num_states, env.num_actions),
        'Expected SARSA': ExpectedSarsaAgent(env.num_states, env.num_actions),
        'Double Q': DoubleQLearningAgent(env.num_states, env.num_actions),
    }

    results = {}
    metrics = {}
    for name, agent in agents.items():
        rewards, xh, uh = run_experiment(agent, env, episodes=1000)
        results[name] = (rewards, xh, uh)
        metrics[name] = evaluate(xh, uh, ref_seq, rewards)

    # Plot Episode Returns
    plt.figure(figsize=(8,4))
    for name, (rew, _, _) in results.items():
        avg_ret = metrics[name][0]
        plt.plot(rew, label=f"{name} (avgR₅₀={avg_ret:.1f})")
    plt.title("Episode Returns")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)

    # Plot Reference vs Response
    plt.figure(figsize=(8,4))
    plt.plot(time, ref_seq, 'k--', label='Reference')
    for name, (_, xh, _) in results.items():
        mse = metrics[name][1]
        plt.plot(time, xh, label=f"{name} (MSE={mse:.2f})")
    plt.title("Reference vs Response")
    plt.xlabel("Time (s)")
    plt.ylabel("x(t)")
    plt.legend()
    plt.grid(True)

    # Plot Control Signals
    plt.figure(figsize=(8,4))
    for name, (_, _, uh) in results.items():
        avg_u = metrics[name][3]
        plt.step(time, uh, where='post', label=f"{name} (ū={avg_u:.2f})")
    plt.title("Control Signals")
    plt.xlabel("Time (s)")
    plt.ylabel("u(t)")
    plt.legend()
    plt.grid(True)

    # Bar-chart of all metrics
    names = list(metrics.keys())
    avg_rets = [metrics[n][0] for n in names]
    mses     = [metrics[n][1] for n in names]
    outs     = [metrics[n][2] for n in names]
    avg_us   = [metrics[n][3] for n in names]

    x = np.arange(len(names))
    width = 0.2

    fig, axs = plt.subplots(2,2, figsize=(10,8))
    axs = axs.flatten()
    axs[0].bar(x-width, avg_rets, width); axs[0].set_title("Avg Return (last 50)")
    axs[1].bar(x, mses, width);       axs[1].set_title("MSE")
    axs[2].bar(x+width, outs, width); axs[2].set_title("% Out of Tol")
    axs[3].bar(x+2*width, avg_us, width); axs[3].set_title("Avg |u|")

    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.grid(True)

    plt.tight_layout()
    plt.show()
