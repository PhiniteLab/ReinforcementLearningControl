import numpy as np
import matplotlib.pyplot as plt

# --- 1) Orijinal Plant Sınıfı (aynı) ---
class FirstOrderPlant:
    def __init__(self, dt: float, ref_seq: np.ndarray,
                 error_bins: np.ndarray, lam: float = 0.01,
                 beta: float = 0.1):
        self.dt = dt
        self.ref_seq = ref_seq
        self.error_bins = error_bins
        self.lam = lam
        self.beta = beta
        self.actions = np.arange(-10, 11)
        self.num_actions = len(self.actions)
        self.num_states = len(error_bins) + 1
        self.prev_u = 0.0

    def discretize_error(self, e: float) -> int:
        idx = np.digitize(e, self.error_bins)
        return int(np.clip(idx, 0, self.num_states - 1))

    def step(self, x: float, t: int, a: int) -> tuple[float, float, int, float]:
        u = self.actions[a]
        delta_u = u - self.prev_u
        self.prev_u = u
        x_next = x + self.dt * (-x + u)
        r_ref = self.ref_seq[t]
        e = x_next - r_ref
        reward = -(1000.0 * e**2 + self.lam * u**2 + self.beta * delta_u**2)
        s_next = self.discretize_error(e)
        return x_next, reward, s_next, u

    def reset(self) -> tuple[float, int, int]:
        x0 = 0.0
        self.prev_u = 0.0
        e0 = x0 - self.ref_seq[0]
        s0 = self.discretize_error(e0)
        return x0, s0, 0

# --- 2) Helper: composite reference ---
def generate_composite_reference(time: np.ndarray, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    ref = (2.0 * np.sin(0.5 * time)
           + 1.5 * np.cos(time)
           + 0.05 * np.random.randn(len(time))
          ) + 5.0
    return np.clip(ref, 0, 10)

# --- 3) DP Model Building (deterministic) ---
def build_time_indexed_model(env: FirstOrderPlant):
    dt = env.dt
    ref = env.ref_seq
    bins = env.error_bins
    actions = env.actions
    lam, beta = env.lam, env.beta

    T = len(ref)
    S_e = env.num_states
    A = env.num_actions
    S = S_e * A  # augmented states: (error_bin, prev_u)

    # Precompute error midpoints
    edges = bins
    mids = np.zeros(S_e)
    for s in range(S_e):
        if s == 0:
            mids[s] = edges[0] - 0.5*(edges[1]-edges[0])
        elif s == S_e-1:
            mids[s] = edges[-1] + 0.5*(edges[-1]-edges[-2])
        else:
            mids[s] = 0.5*(edges[s-1]+edges[s])

    # Allocate deterministic model
    P = np.zeros((T, S, A), dtype=int)   # P[t,s,a] = next_state index
    R = np.zeros((T, S, A), dtype=float) # R[t,s,a] = reward

    for t in range(T):
        for s_idx in range(S):
            s_e = s_idx // A
            prev_u_idx = s_idx % A
            prev_u = actions[prev_u_idx]
            # reconstruct x_t from ref and error midpoint
            x = ref[t] + mids[s_e]
            for a_idx, u in enumerate(actions):
                # dynamics and reward
                x_next = x + dt * (-x + u)
                e_next = x_next - ref[t]
                delta_u = u - prev_u
                r = -(1000.0*e_next**2 + lam*u**2 + beta*delta_u**2)
                s_e_next = int(np.clip(np.digitize(e_next, bins), 0, S_e-1))
                next_idx = s_e_next * A + a_idx
                P[t, s_idx, a_idx] = next_idx
                R[t, s_idx, a_idx] = r

    return P, R, T, S, A

# --- 4) Finite‐Horizon Policy Iteration ---
def finite_horizon_policy_iteration(P, R, T: int, S: int, A: int):
    # V[T,·]=0 initial, π arbitrary
    V = np.zeros((T+1, S))
    policy = np.zeros((T, S), dtype=int)

    # initialize policy randomly
    policy[:] = np.random.randint(A)

    # policy iteration
    stable = False
    while not stable:
        # (a) Policy Evaluation (backward)
        for t in reversed(range(T)):
            for s in range(S):
                a = policy[t, s]
                s2 = P[t, s, a]
                V[t, s] = R[t, s, a] + V[t+1, s2]

        # (b) Policy Improvement
        stable = True
        for t in range(T):
            for s in range(S):
                # compute Q for all actions
                q_vals = np.array([
                    R[t, s, a] + V[t+1, P[t, s, a]]
                    for a in range(A)
                ])
                best_a = int(np.argmax(q_vals))
                if best_a != policy[t, s]:
                    stable = False
                    policy[t, s] = best_a

    return policy, V

# --- 5) Simulation with π_t(s) ---
def simulate(env: FirstOrderPlant, policy, T: int):
    dt = env.dt
    ref = env.ref_seq
    bins = env.error_bins
    actions = env.actions
    A = env.num_actions
    S_e = env.num_states
    S = S_e * A

    # recompute mids
    edges = bins
    mids = np.zeros(S_e)
    for s in range(S_e):
        if s == 0:
            mids[s] = edges[0] - 0.5*(edges[1]-edges[0])
        elif s == S_e-1:
            mids[s] = edges[-1] + 0.5*(edges[-1]-edges[-2])
        else:
            mids[s] = 0.5*(edges[s-1]+edges[s])

    x = 0.0
    prev_u = 0.0
    prev_u_idx = int(np.where(actions==prev_u)[0])
    x_hist = np.zeros(T)
    u_hist = np.zeros(T)

    for t in range(T):
        e = x - ref[t]
        s_e = int(np.clip(np.digitize(e, bins), 0, S_e-1))
        s_idx = s_e * A + prev_u_idx
        a_idx = policy[t, s_idx]
        u = actions[a_idx]
        # apply plant step
        x = x + dt * (-x + u)
        x_hist[t] = x
        u_hist[t] = u
        prev_u_idx = a_idx

    return x_hist, u_hist

# --- 6) Plotting ---
def plot_results(time, ref_seq, x_hist, u_hist):
    fig, axs = plt.subplots(3,1, figsize=(8,10))

    axs[0].plot(time, ref_seq, '--', label='Reference')
    axs[0].plot(time, x_hist, label='Response')
    axs[0].set(title='Reference vs Response', xlabel='Time (s)', ylabel='x(t)')
    axs[0].legend(); axs[0].grid(True)

    axs[1].step(time, u_hist, where='post')
    axs[1].set(title='Control Signal u(t)', xlabel='Time (s)', ylabel='u')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# --- Main ---
if __name__ == "__main__":
    # parameters
    dt = 0.01
    total_time = 5.0
    time = np.arange(0, total_time, dt)
    ref_seq = generate_composite_reference(time, seed=42)
    error_bins = np.linspace(-5, 5, 21)

    # init env
    env = FirstOrderPlant(dt, ref_seq, error_bins, lam=0.001, beta=1.0)

    # build model
    P, R, T, S, A = build_time_indexed_model(env)

    # policy iteration
    policy, V = finite_horizon_policy_iteration(P, R, T, S, A)

    # simulate
    x_hist, u_hist = simulate(env, policy, T)

    # plot
    plot_results(time, ref_seq, x_hist, u_hist)
