import numpy as np
import matplotlib.pyplot as plt

# --- Environment ---
class FirstOrderPlant:
    def __init__(self, dt, ref_seq, x_bins, ref_bins, error_bins, error_weight=10.0, lam=0.01, beta=0.1):
        self.dt = dt
        self.ref_seq = ref_seq
        self.x_bins = x_bins
        self.ref_bins = ref_bins
        self.error_bins = error_bins
        self.error_weight = error_weight
        self.lam = lam
        self.beta = beta
        self.actions = np.arange(-10, 11, 0.5)
        self.num_actions = len(self.actions)
        self.num_x = len(x_bins) + 1
        self.num_ref = len(ref_bins) + 1
        self.num_error = len(error_bins) + 1
        self.prev_u = 0.0

    def discretize_state(self, x, r):
        x_idx = np.digitize(x, self.x_bins)
        r_idx = np.digitize(r, self.ref_bins)
        e = x - r
        e_idx = np.digitize(e, self.error_bins)
        return (x_idx, r_idx, e_idx)

    def flat_state(self, s_tuple):
        x_idx, r_idx, e_idx = s_tuple
        return x_idx * self.num_ref * self.num_error + r_idx * self.num_error + e_idx

    def step(self, x, t, a):
        u = self.actions[a]
        delta_u = u - self.prev_u
        self.prev_u = u
        x_next = x + self.dt * (-x + u)
        r = self.ref_seq[t]
        e = x_next - r
        reward = -(
            self.error_weight * e ** 2 +
            self.lam * u ** 2 +
            self.beta * delta_u ** 2
        )
        s_next_tuple = self.discretize_state(x_next, r)
        s_next = self.flat_state(s_next_tuple)
        return x_next, reward, s_next, u

    def reset(self):
        x0 = 0.0
        self.prev_u = 0.0
        r0 = self.ref_seq[0]
        s0_tuple = self.discretize_state(x0, r0)
        s0 = self.flat_state(s0_tuple)
        return x0, s0, 0

# --- n-step SARSA (on-policy, batch/episode-style) ---
class NStepSarsaAgent:
    def __init__(self, S, A, n=5, gamma=0.99, alpha=0.05,
                 epsilon=1.0, min_eps=0.01, decay=0.995):
        self.S = S
        self.A = A
        self.n = n
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.decay = decay
        self.Q = np.zeros((S, A))

    def select(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.A)
        return int(np.argmax(self.Q[s]))

    def update_online(self, memory):
        T = len(memory)
        # Sliding window yok: Her t için bir kez, (t, t+1, ..., t+n) episode sonunda hesaplanır.
        for t in range(T):
            G = 0.0
            discount = 1.0
            for k in range(self.n):
                if t + k < T:
                    G += discount * memory[t + k][2]
                    discount *= self.gamma
                else:
                    break
            if t + self.n < T:
                s_next, a_next = memory[t + self.n][0], memory[t + self.n][1]
                G += discount * self.Q[s_next, a_next]
            # else terminal; sadece ödül toplamı
            s, a = memory[t][0], memory[t][1]
            self.Q[s, a] += self.alpha * (G - self.Q[s, a])
        self.epsilon = max(self.epsilon * self.decay, self.min_eps)

# --- n-step SARSA (off-policy, batch/episode-style) ---
class NStepSarsaOffPolicyAgent:
    def __init__(self, S, A, n=5, gamma=0.99, alpha=0.05,
                 epsilon=1.0, min_eps=0.01, decay=0.995):
        self.S = S
        self.A = A
        self.n = n
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.decay = decay
        self.Q = np.zeros((S, A))

    def select(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.A)
        return int(np.argmax(self.Q[s]))

    def update_online(self, memory, behavior_probs):
        T = len(memory)
        for t in range(T):
            G = 0.0
            discount = 1.0
            rho = 1.0
            for k in range(self.n):
                if t + k < T:
                    G += discount * memory[t + k][2]
                    discount *= self.gamma
                    greedy_a = np.argmax(self.Q[memory[t + k][0]])
                    pi_a = 1.0 if memory[t + k][1] == greedy_a else 0.0
                    mu_a = max(behavior_probs[t + k], 1e-6)
                    if mu_a > 0:
                        rho *= pi_a / mu_a
                else:
                    break
            if t + self.n < T:
                s_next, a_next = memory[t + self.n][0], np.argmax(self.Q[memory[t + self.n][0]])
                G += discount * self.Q[s_next, a_next]
            # else terminal
            s, a = memory[t][0], memory[t][1]
            self.Q[s, a] += self.alpha * rho * (G - self.Q[s, a])
        self.epsilon = max(self.epsilon * self.decay, self.min_eps)

# --- One-step SARSA Agent ---
class OneStepSarsaAgent:
    def __init__(self, S, A, gamma=0.99, alpha=0.1,
                 epsilon=1.0, min_eps=0.01, decay=0.995):
        self.S = S
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.decay = decay
        self.Q = np.zeros((S, A))

    def select(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.A)
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, s2, a2):
        target = r + self.gamma * self.Q[s2, a2]
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        self.epsilon = max(self.epsilon * self.decay, self.min_eps)
"""
# --- Reference generator ---
def generate_composite_reference(time, seed=None):
    if seed is not None:
        np.random.seed(seed)

    sin1 = 1.0 * np.sin(0.5 * time + np.random.uniform(0, 2*np.pi))
    sin2 = 0.7 * np.sin(2.1 * time + np.random.uniform(0, 2*np.pi))
    sin3 = 0.4 * np.sin(4.3 * time + np.random.uniform(0, 2*np.pi))
    ramp = 0.25 * time
    step = np.zeros_like(time)
    step[time > 2.0] += 1.5
    step[time > 5.5] -= 2.0
    step[time > 7.0] += 1.0
    tri = 0.8 * (2 * np.abs(2 * ((time/5.0) % 1) - 1) - 1)
    rect = 1.0 * (np.sign(np.sin(1.1 * time)) + 1) / 2
    pulse = np.zeros_like(time)
    pulse[(time > 1.0) & (time < 1.2)] += 2.0
    pulse[(time > 8.0) & (time < 8.1)] -= 1.5
    noise = 0.15 * np.random.randn(len(time))

    ref = 2.0 + sin1 + sin2 + sin3 + ramp + step + tri + rect + pulse + noise
    ref = np.clip(ref, 0, 10)
    return ref

"""
# --- Reference generator ---
def generate_composite_reference(time, seed=None):
    if seed is not None: np.random.seed(seed)
    ref = 2 * np.sin(0.5 * time) + 1.5 * np.cos(time) + 0.05 * np.random.randn(len(time))
    return np.clip(ref + 5, 0, 10)

# --- Run experiment ---
def run_experiment(agent, env, episodes=500, off_policy=False, one_step=False):
    T = len(env.ref_seq)
    rewards = []
    for ep in range(episodes):
        x, s, t = env.reset()
        traj = []
        behavior_probs = []
        total_r = 0.0
        if one_step:
            a = agent.select(s)
            for _ in range(T):
                x2, r, s2, _ = env.step(x, t, a)
                a2 = agent.select(s2)
                agent.update(s, a, r, s2, a2)
                x, s, a, t = x2, s2, a2, t+1
                total_r += r
        else:
            for i in range(T):
                greedy = np.argmax(agent.Q[s])
                if np.random.rand() < agent.epsilon:
                    a = np.random.randint(env.num_actions)
                else:
                    a = greedy
                mu = agent.epsilon / env.num_actions + (1 - agent.epsilon) if a == greedy else agent.epsilon / env.num_actions
                behavior_probs.append(mu)
                x2, r, s2, u = env.step(x, t, a)
                traj.append((s, a, r, s2, u))
                x, s, t = x2, s2, t+1
                total_r += r
            if off_policy:
                agent.update_online(traj, behavior_probs)
            else:
                agent.update_online(traj)
        rewards.append(total_r)

    # simulate final policy
    x = 0.0
    x_hist = np.zeros(T)
    u_hist = np.zeros(T)
    env.reset()
    for t in range(T):
        r = env.ref_seq[t]
        s_tuple = env.discretize_state(x, r)
        s = env.flat_state(s_tuple)
        a = int(np.argmax(agent.Q[s]))
        x, _, _, u = env.step(x, t, a)
        x_hist[t] = x
        u_hist[t] = u
    return np.array(rewards), x_hist, u_hist

# --- Evaluation metrics ---
def evaluate(x_hist, u_hist, ref_seq, returns, last_N=50):
    mse = np.mean((x_hist - ref_seq) ** 2)
    tol = 0.1
    out_of_tol = 100 * np.mean(np.abs(x_hist - ref_seq) > tol)
    avg_u = np.mean(np.abs(u_hist))
    avg_ret = np.mean(returns[-last_N:])
    return avg_ret, mse, out_of_tol, avg_u

# --- Main & Plotting ---
if __name__ == "__main__":
    dt = 0.01
    total_time = 10.0
    time = np.arange(0, total_time, dt)
    ref_seq = generate_composite_reference(time, seed=42)

    x_bins = np.linspace(0, 10, 41)
    ref_bins = np.linspace(0, 10, 41)
    error_bins = np.linspace(-5, 5, 41)
    env = FirstOrderPlant(dt, ref_seq, x_bins, ref_bins, error_bins,
                          error_weight=100000.0, lam=0.01, beta=10.0)

    num_states = (len(x_bins)+1)*(len(ref_bins)+1)*(len(error_bins)+1)

    agents = {
        'One-step SARSA (n=1)': OneStepSarsaAgent(num_states, env.num_actions),
        'n-Step SARSA (on-policy, n=20)': NStepSarsaAgent(num_states, env.num_actions, n=5, alpha=0.4),
        'n-Step SARSA (off-policy, n=20)': NStepSarsaOffPolicyAgent(num_states, env.num_actions, n=5, alpha=0.4),
    }

    num_episodes = 5000
    results = {}
    metrics = {}
    for name, agent in agents.items():
        if name.startswith('One-step'):
            rewards, xh, uh = run_experiment(agent, env, episodes=num_episodes, one_step=True)
        elif 'off-policy' in name:
            rewards, xh, uh = run_experiment(agent, env, episodes=num_episodes, off_policy=True)
        else:
            rewards, xh, uh = run_experiment(agent, env, episodes=num_episodes)
        results[name] = (rewards, xh, uh)
        metrics[name] = evaluate(xh, uh, ref_seq, rewards)

    # Plot Episode Returns
    plt.figure(figsize=(8, 4))
    for name, (rew, _, _) in results.items():
        avg_ret = metrics[name][0]
        plt.plot(rew, label=f"{name} (avgR₅₀={avg_ret:.1f})")
    plt.title("Episode Returns")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)

    # Plot Reference vs Response
    plt.figure(figsize=(8, 4))
    plt.plot(time, ref_seq, 'k--', label='Reference')
    for name, (_, xh, _) in results.items():
        mse = metrics[name][1]
        plt.plot(time, xh, label=f"{name} (MSE={mse:.2f})")
    plt.title("Reference vs Response")
    plt.xlabel("Time (s)")
    plt.ylabel("x(t)")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 8))
    n_methods = len(results)
    names = list(results.keys())
    for i, name in enumerate(names):
        _, _, uh = results[name]
        avg_u = metrics[name][3]
        plt.subplot(n_methods, 1, i + 1)
        plt.step(time, uh, where='post', label=f"{name} (ū={avg_u:.2f})")
        plt.ylabel("u(t)")
        plt.legend()
        plt.grid(True)
        if i == 0:
            plt.title("Control Signals")
        if i < n_methods - 1:
            plt.xticks([])

    plt.xlabel("Time (s)")
    plt.tight_layout()

    # Bar-chart of all metrics
    names = list(metrics.keys())
    avg_rets = [metrics[n][0] for n in names]
    mses = [metrics[n][1] for n in names]
    outs = [metrics[n][2] for n in names]
    avg_us = [metrics[n][3] for n in names]

    x = np.arange(len(names))
    width = 0.2

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    axs[0].bar(x - width, avg_rets, width); axs[0].set_title("Avg Return (last 50)")
    axs[1].bar(x, mses, width);       axs[1].set_title("MSE")
    axs[2].bar(x + width, outs, width); axs[2].set_title("% Out of Tol")
    axs[3].bar(x + 2 * width, avg_us, width); axs[3].set_title("Avg |u|")

    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.grid(True)

    plt.tight_layout()
    plt.show()
