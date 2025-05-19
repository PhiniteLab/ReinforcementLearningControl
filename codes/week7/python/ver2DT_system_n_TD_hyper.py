import numpy as np
import matplotlib.pyplot as plt
import itertools

# --- Environment ---
class FirstOrderPlant:
    def __init__(self, dt, ref_seq, error_bins, error_weight=10.0, lam=0.01, beta=0.1):
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

    def discretize_error(self, e):
        idx = np.digitize(e, self.error_bins)
        return int(np.clip(idx, 0, self.num_states - 1))

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
        s_next = self.discretize_error(e)
        return x_next, reward, s_next, u

    def reset(self):
        x0 = 0.0
        self.prev_u = 0.0
        e0 = x0 - self.ref_seq[0]
        s0 = self.discretize_error(e0)
        return x0, s0, 0

# --- One-step SARSA Agent ---
class OneStepSarsaAgent:
    def __init__(self, S, A, gamma=0.99, alpha=0.1,
                 epsilon=0.1, min_eps=0.01, decay=0.995):
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

# --- n-step SARSA (on-policy, sliding window) ---
class NStepSarsaAgent:
    def __init__(self, S, A, n=5, gamma=0.99, alpha=0.05,
                 epsilon=0.1, min_eps=0.01, decay=0.995):
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
        for t in range(T - self.n):
            G = 0.0
            discount = 1.0
            for k in range(self.n):
                G += discount * memory[t + k][2]
                discount *= self.gamma
            s_next, a_next = memory[t + self.n][0], memory[t + self.n][1]
            G += discount * self.Q[s_next, a_next]
            s, a = memory[t][0], memory[t][1]
            self.Q[s, a] += self.alpha * (G - self.Q[s, a])
        # Kalan son adımlar (bootstrapping olmadan)
        for t in range(T - self.n, T):
            G = 0.0
            discount = 1.0
            for k in range(self.n):
                if t + k < T:
                    G += discount * memory[t + k][2]
                    discount *= self.gamma
            s, a = memory[t][0], memory[t][1]
            self.Q[s, a] += self.alpha * (G - self.Q[s, a])
        self.epsilon = max(self.epsilon * self.decay, self.min_eps)

# --- n-step SARSA (off-policy, importance sampling, sliding window) ---
class NStepSarsaOffPolicyAgent:
    def __init__(self, S, A, n=5, gamma=0.99, alpha=0.05,
                 epsilon=0.1, min_eps=0.01, decay=0.995):
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
        for t in range(T - self.n):
            G = 0.0
            discount = 1.0
            rho = 1.0
            for k in range(self.n):
                G += discount * memory[t + k][2]
                discount *= self.gamma
                greedy_a = np.argmax(self.Q[memory[t + k][0]])
                pi_a = 1.0 if memory[t + k][1] == greedy_a else 0.0
                mu_a = max(behavior_probs[t + k], 1e-6)
                if mu_a > 0:
                    rho *= pi_a / mu_a
            s_next, a_next = memory[t + self.n][0], np.argmax(self.Q[memory[t + self.n][0]])
            G += discount * self.Q[s_next, a_next]
            s, a = memory[t][0], memory[t][1]
            self.Q[s, a] += self.alpha * rho * (G - self.Q[s, a])
        # Kalan son adımlar (bootstrapping olmadan)
        for t in range(T - self.n, T):
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
            s, a = memory[t][0], memory[t][1]
            self.Q[s, a] += self.alpha * rho * (G - self.Q[s, a])
        self.epsilon = max(self.epsilon * self.decay, self.min_eps)

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
        s = env.discretize_error(x - env.ref_seq[t])
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

def hyperparam_grid_search(agent_class, param_grid, env, episodes=300, off_policy=False, one_step=False):
    keys, values = zip(*param_grid.items())
    combos = list(itertools.product(*values))
    total_configs = len(combos)
    best_score = -np.inf  # maximize avg return
    best_params = None
    best_metrics = None
    best_Q = None

    print(f"\n[Hyperparameter Search] Total configurations: {total_configs}")

    for i, combination in enumerate(combos):
        params = dict(zip(keys, combination))
        agent = agent_class(env.num_states, env.num_actions, **params)
        rewards, xh, uh = run_experiment(agent, env, episodes=episodes, off_policy=off_policy, one_step=one_step)
        avg_ret, mse, out_tol, avg_u = evaluate(xh, uh, env.ref_seq, rewards)
        score = avg_ret  # Alternatif: score = -mse

        print(f"[{i+1}/{total_configs}] Params: {params} | MSE: {mse:.4f} | AvgRet: {avg_ret:.2f}")

        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_metrics = (avg_ret, mse, out_tol, avg_u)
            best_Q = agent.Q.copy()

    print(f"\n[Hyperparam Search Done] Best Params: {best_params} | Best Metrics (AvgRet, MSE, %OutTol, Avg|u|): {best_metrics}\n")
    return best_params, best_metrics, best_Q

# --- MAIN & HYPERPARAM OPTIMIZATION ---
if __name__ == "__main__":
    dt = 0.01
    total_time = 10.0
    time = np.arange(0, total_time, dt)
    ref_seq = generate_composite_reference(time, seed=42)
    error_bins = np.linspace(-5, 5, 21)
    env = FirstOrderPlant(dt, ref_seq, error_bins, error_weight=10000.0, lam=0.01, beta=10.0)

    # Hiperparametre grid'leri
    grid_onestep = {
        "alpha": [0.02, 0.05, 0.1, 0.2],
        "epsilon": [0.05, 0.1, 0.2, 0.3],
        "gamma": [0.95, 0.99],
        "decay": [0.995, 0.999],
    }
    grid_nstep = {
        "alpha": [0.005, 0.05, 0.1, 0.2],
        "epsilon": [0.1, 0.5, 1.0],
        "gamma": [0.5, 0.8, 0.99],
        "decay": [0.995],
        "n": [2, 3, 5, 7, 10],
    }

    print("\nHyperparameter search: One-step SARSA...")
    best_params_1, best_metrics_1, best_Q_1 = hyperparam_grid_search(
        OneStepSarsaAgent, grid_onestep, env, episodes=100, one_step=True
    )
    print("Best One-step SARSA:", best_params_1, best_metrics_1)

    print("\nHyperparameter search: n-step SARSA (on-policy)...")
    best_params_2, best_metrics_2, best_Q_2 = hyperparam_grid_search(
        NStepSarsaAgent, grid_nstep, env, episodes=100
    )
    print("Best n-step SARSA:", best_params_2, best_metrics_2)

    print("\nHyperparameter search: n-step SARSA (off-policy)...")
    best_params_3, best_metrics_3, best_Q_3 = hyperparam_grid_search(
        NStepSarsaOffPolicyAgent, grid_nstep, env, episodes=100, off_policy=True
    )
    print("Best n-step SARSA (off-policy):", best_params_3, best_metrics_3)

    # EN İYİ parametrelerle yeniden uzun (final) deney
    agents = {
        f"One-step SARSA\n{best_params_1}": OneStepSarsaAgent(env.num_states, env.num_actions, **best_params_1),
        f"n-step SARSA (on-policy)\n{best_params_2}": NStepSarsaAgent(env.num_states, env.num_actions, **best_params_2),
        f"n-step SARSA (off-policy)\n{best_params_3}": NStepSarsaOffPolicyAgent(env.num_states, env.num_actions, **best_params_3),
    }

    results = {}
    metrics = {}
    for name, agent in agents.items():
        if "One-step" in name:
            rewards, xh, uh = run_experiment(agent, env, episodes=10000, one_step=True)
        elif "off-policy" in name:
            rewards, xh, uh = run_experiment(agent, env, episodes=10000, off_policy=True)
        else:
            rewards, xh, uh = run_experiment(agent, env, episodes=10000)
        results[name] = (rewards, xh, uh)
        metrics[name] = evaluate(xh, uh, ref_seq, rewards)

    # Plot Episode Returns
    plt.figure(figsize=(8, 4))
    for name, (rew, _, _) in results.items():
        avg_ret = metrics[name][0]
        plt.plot(rew, label=f"{name.splitlines()[0]} (avgR₅₀={avg_ret:.1f})")
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
        plt.plot(time, xh, label=f"{name.splitlines()[0]} (MSE={mse:.2f})")
    plt.title("Reference vs Response")
    plt.xlabel("Time (s)")
    plt.ylabel("x(t)")
    plt.legend()
    plt.grid(True)

    # Plot Control Signals
    plt.figure(figsize=(8, 4))
    for name, (_, _, uh) in results.items():
        avg_u = metrics[name][3]
        plt.step(time, uh, where='post', label=f"{name.splitlines()[0]} (ū={avg_u:.2f})")
    plt.title("Control Signals")
    plt.xlabel("Time (s)")
    plt.ylabel("u(t)")
    plt.legend()
    plt.grid(True)

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
        ax.set_xticklabels([n.splitlines()[0] for n in names], rotation=15)
        ax.grid(True)

    plt.tight_layout()
    plt.show()
