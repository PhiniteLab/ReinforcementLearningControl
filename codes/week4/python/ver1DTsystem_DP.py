import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Dynamic Programming Controller for FirstOrderPlant
# Implements finite-horizon backward induction (Policy Iteration)
# based on Bellman equation (see Chapter 4)
# ---------------------------------------------------

dt = 0.05
T_total = 5.0
time = np.arange(0, T_total, dt)
N = len(time)

def generate_composite_reference(time: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Create a dynamic reference sequence:
    ref[t] = 2 sin(0.5 t) + 1.5 cos(t) + noise + 5
    """
    if seed is not None:
        np.random.seed(seed)
    ref = (
        2.0 * np.sin(0.5 * time)
        + 1.5 * np.cos(time)
        + 0.05 * np.random.randn(len(time))
        + 5.0
    )
    return np.clip(ref, 0.0, 10.0)

# Plant and reward parameters
ref_seq = generate_composite_reference(time, seed=42)
lam = 0.001    # control effort weight
beta = 0.1     # delta_u penalty

actions = np.arange(-10, 11)    # discrete control torques
A = len(actions)

# Discretize error into bins
error_bins = np.linspace(-5, 5, 21)
S_e = len(error_bins) + 1

# Representative midpoint for each error state
error_mids = np.zeros(S_e)
edges = error_bins
for s in range(S_e):
    if s == 0:
        error_mids[s] = edges[0] - 0.5*(edges[1]-edges[0])
    elif s == S_e-1:
        error_mids[s] = edges[-1] + 0.5*(edges[-1]-edges[-2])
    else:
        error_mids[s] = 0.5*(edges[s-1]+edges[s])

# --------------
# Helper function
# --------------
def discretize_error(e: float) -> int:
    """Map continuous error e to discrete state index"""
    idx = np.digitize(e, error_bins)
    return int(np.clip(idx, 0, S_e - 1))

# ------------------------------------------------------
# 1) DP Backward Induction (Finite-Horizon)
# Bellman eqn (1): V_t(s) = max_a [ r(s,a) + V_{t+1}(s') ]
# No discounting (gamma=1)
# State index s_idx = error_state * A + prev_action_index
# ------------------------------------------------------
gamma = 1.0
S = S_e * A       # total augmented states (error x prev_u)
V_next = np.zeros(S)  # V_{t+1}(s)
policy = np.zeros((N, S), dtype=int)

for t in reversed(range(N)):
    V_t = np.empty(S)
    for s_idx in range(S):
        s_e = s_idx // A            # error state
        prev_u_idx = s_idx % A      # index of previous u
        prev_u = actions[prev_u_idx]
        # reconstruct plant output: x = ref + error
        x = ref_seq[t] + error_mids[s_e]

        best_val = -np.inf
        best_a = 0
        for a_idx, u in enumerate(actions):
            # step eqn: x_{t+1} = x + dt(-x + u)
            x_next = x + dt * (-x + u)
            e_next = x_next - ref_seq[t]
            delta_u = u - prev_u
            # reward = -[10 e^2 + lam u^2 + beta (delta_u)^2]
            r = -(10*e_next**2 + lam*u**2 + beta*delta_u**2)
            s_e_next = discretize_error(e_next)
            next_idx = s_e_next * A + a_idx
            # Bellman backup
            val = r + gamma * V_next[next_idx]
            if val > best_val:
                best_val = val
                best_a = a_idx
        V_t[s_idx] = best_val
        policy[t, s_idx] = best_a
    V_next = V_t

# ------------------------------------------------------
# 2) Simulation using DP policy
#    x_{t+1} = x_t + dt(-x_t + u_t)
# ------------------------------------------------------
x = 0.0
prev_u = 0.0
x_hist = np.zeros(N)
u_hist = np.zeros(N)
for t in range(N):
    e = x - ref_seq[t]
    s_e = discretize_error(e)
    prev_u_idx = int(np.where(actions==prev_u)[0])
    s_idx = s_e * A + prev_u_idx
    a_idx = policy[t, s_idx]
    u = actions[a_idx]
    # apply plant
    x = x + dt * (-x + u)
    x_hist[t] = x
    u_hist[t] = u
    prev_u = u

# ------------------------------------------------------
# 3) Plot results
# ------------------------------------------------------
fig, axs = plt.subplots(3,1,figsize=(8,10))
# Reference vs response
axs[0].plot(time, ref_seq,'--',label='Reference')
axs[0].plot(time, x_hist, label='Response')
axs[0].set(title='Reference vs Response', xlabel='Time (s)', ylabel='x(t)')
axs[0].legend(); axs[0].grid(True)

# Control signal
axs[1].step(time, u_hist, where='post')
axs[1].set(title='Control Signal u(t)', xlabel='Time (s)', ylabel='u'); axs[1].grid(True)

# Initial policy at t=0: error vs optimal u
best_actions_t0 = policy[0].reshape(S_e,A).argmax(axis=1)
axs[2].step(error_mids, actions[best_actions_t0], where='mid')
axs[2].set(title='DP Policy at t=0', xlabel='Error', ylabel='u*(error)')
axs[2].grid(True)

plt.tight_layout()
plt.show()
