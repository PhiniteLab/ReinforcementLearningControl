import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
from itertools import product

# --- Environment and Agent definitions ---
class FirstOrderPlant:
    def __init__(self, dt: float, ref_seq: np.ndarray, error_bins: np.ndarray, lam: float = 0.01):
        self.dt = dt
        self.ref_seq = ref_seq
        self.error_bins = error_bins
        self.lam = lam
        self.actions = np.arange(-10, 11)
        self.num_actions = len(self.actions)
        self.num_states = len(error_bins) + 1

    def discretize_error(self, e: float) -> int:
        idx = np.digitize(e, self.error_bins)
        return int(np.clip(idx, 0, self.num_states - 1))

    def step(self, x, t, a):
        u = self.actions[a]
        x_next = x + self.dt * (-x + u)
        r = self.ref_seq[t]
        e = x_next - r
        reward = -(e**2 + self.lam * u**2)
        s_next = self.discretize_error(e)
        return x_next, reward, s_next, u

    def reset(self):
        x0 = 0.0
        e0 = x0 - self.ref_seq[0]
        s0 = self.discretize_error(e0)
        return x0, s0, 0

class QLearningAgent:
    def __init__(self, num_states, num_actions, epsilon, alpha, gamma):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((num_states, num_actions))

    def select_action(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, s_next):
        target = r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

def train_agent(env, agent, episodes):
    T = len(env.ref_seq)
    rewards = []
    for _ in range(episodes):
        x, s, t = env.reset()
        total = 0.0
        for _ in range(T):
            a = agent.select_action(s)
            x, r, s_next, _ = env.step(x, t, a)
            agent.update(s, a, r, s_next)
            s, t = s_next, t + 1
            total += r
        rewards.append(total)
    return rewards

def simulate_policy(env, agent):
    T = len(env.ref_seq)
    x = 0.0
    xh, uh = np.zeros(T), np.zeros(T)
    for t in range(T):
        s = env.discretize_error(x - env.ref_seq[t])
        a = int(np.argmax(agent.Q[s]))
        u = env.actions[a]
        x = x + env.dt * (-x + u)
        xh[t], uh[t] = x, u
    return np.arange(T)*env.dt, xh, uh

# --- Setup --- #
dt = 0.01
total_time = 15.0
times = np.arange(0, total_time, dt)
ref_seq = np.clip(2*np.sin(0.5*times) + 1.5*np.cos(1.0*times) + 0.1*np.random.randn(len(times)) + 5, 0, 10)
error_bins = np.linspace(-5, 5, 21)

# Hyperparameter grid
grid = {
    'epsilon': [0.1, 0.001, 0.00001],
    'alpha': [0.05, 0.1, 0.2],
    'gamma': [0.9],
    'lam': [0.01]
}

# Tuning

config_count = np.prod([len(v) for v in grid.values()])
print(f"Tuning {config_count} configurations...")

param_list = list(product(grid['epsilon'], grid['alpha'], grid['gamma'], grid['lam']))

results = []
episodes_tune = 400

for idx, (eps, al, gm, lm) in enumerate(param_list, start=1):
    env_t = FirstOrderPlant(dt, ref_seq, error_bins, lam=lm)
    ag_t  = QLearningAgent(env_t.num_states, env_t.num_actions, eps, al, gm)
    rew   = train_agent(env_t, ag_t, episodes_tune)
    score = np.mean(rew[-50:])
    results.append({
        'epsilon': eps,
        'alpha':   al,
        'gamma':   gm,
        'lam':     lm,
        'score':   score
    })
    # print progress
    print(f"[{idx}/{config_count}] ε={eps:.2f}, α={al:.2f}, γ={gm:.3f}, λ={lm:.2f} → avg_reward={score:.1f}")

df = pd.DataFrame(results)
print(f"Completed tuning.\nTotal configurations: {config_count}\n")
top5 = df.nlargest(5, 'score')
print("Top 5 hyperparameter sets (avg reward):")
print(top5.to_string(index=False))

# Retrain best and compute RMSE
best = top5.iloc[0]
print(f"\nRetraining best config: ε={best.epsilon}, α={best.alpha}, γ={best.gamma}, λ={best.lam}, score={best.score:.1f}\n")
env_b = FirstOrderPlant(dt, ref_seq, error_bins, lam=best.lam)
ag_b = QLearningAgent(env_b.num_states, env_b.num_actions, best.epsilon, best.alpha, best.gamma)
r_full = train_agent(env_b, ag_b, episodes=2000)
time, xh, uh = simulate_policy(env_b, ag_b)
rmse = np.sqrt(np.mean((xh - ref_seq)**2))
print(f"Tracking RMSE (response vs reference): {rmse:.3f}")

# Plot training rewards
plt.figure(figsize=(6,4))
plt.plot(r_full)
plt.title("Training Rewards (Best Config)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()

# Plot tracking and control
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(6,8))
ax1.plot(times, ref_seq, '--', label='Reference')
ax1.plot(time, xh, label='Response')
ax1.set(title="Reference vs Response", xlabel="Time (s)", ylabel="x(t)")
ax1.legend(); ax1.grid(True)
ax2.step(time, uh, where='post')
ax2.set(title="Control Signal u(t)", xlabel="Time (s)", ylabel="u(t)")
ax2.grid(True)
plt.tight_layout()
plt.show()
