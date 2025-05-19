import numpy as np
import matplotlib.pyplot as plt

# --- Plant with reference tracking ---
class FirstOrderPlantWithRef:
    def __init__(self, dt=0.05, noise_std=0.0, ref_func=None):
        self.dt = dt
        self.noise_std = noise_std
        self.ref_func = ref_func if ref_func is not None else (lambda t, ep: 0.0)

    def step(self, x, u):
        # Basit first‐order: x_{t+1} = x + dt*(-x + u)
        return x + self.dt * (-x + u)

    def reward(self, x, u, r):
        e = x - r
        return -(0.1*e*e + 0.1*u*u) * self.dt


# --- Bandit for multi‐parameter arms (Kp, Ki, Kd) ---
class MultiArmedBandit:
    def __init__(self, arms, epsilon=0.1, alpha=None):
        """
        arms: list of (Kp, Ki, Kd) tuples
        if alpha=None → sample‐average update; else constant‐step α update
        """
        self.arms = arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(len(arms), dtype=float)
        self.N = np.zeros(len(arms), dtype=int)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.arms))
        best = self.Q.max()
        choices = np.flatnonzero(self.Q == best)
        return np.random.choice(choices)

    def update(self, a, reward):
        if self.alpha is None:
            # sample‐average
            self.N[a] += 1
            self.Q[a] += (reward - self.Q[a]) / self.N[a]
        else:
            # constant‐step
            self.Q[a] += self.alpha * (reward - self.Q[a])


# --- Run an experiment (now PID) ---
def run_pid_tuning(agent, env, episodes, ep_length):
    rewards = np.zeros(episodes)
    for ep in range(episodes):
        a = agent.select_action()
        Kp, Ki, Kd = agent.arms[a]

        x = 0.0
        integral = 0.0
        e_prev = 0.0
        total_r = 0.0

        for t in range(ep_length):
            r = env.ref_func(t, ep)
            e = x - r
            integral += e * env.dt
            derivative = (e - e_prev) / env.dt if t>0 else 0.0

            # PID kontrol sinyali
            u = -Kp*e - Ki*integral - Kd*derivative

            x = env.step(x, u)
            total_r += env.reward(x, u, r)

            e_prev = e

        agent.update(a, total_r)
        rewards[ep] = total_r

    return rewards


if __name__ == '__main__':
    np.random.seed(0)

    # 1) Discrete grid of (Kp, Ki, Kd)
    Kp_vals = np.linspace(0.5, 10.0, 6)
    Ki_vals = np.linspace(0.1, 10.0, 6)
    Kd_vals = np.linspace(0.01, 0.5, 6)
    arms = [(kp, ki, kd) for kp in Kp_vals for ki in Ki_vals for kd in Kd_vals]

    # 2) Constant step reference
    ref_func = lambda t, ep: 1.0

    env = FirstOrderPlantWithRef(dt=0.01, noise_std=0.0, ref_func=ref_func)
    agent_pid_stat = MultiArmedBandit(arms, epsilon=0.1, alpha=None)
    agent_pid_non  = MultiArmedBandit(arms, epsilon=0.1, alpha=0.1)

    episodes, ep_length = 300, 2000  # how many times iterate algorithm, how many times simulation runs
    rew_stat = run_pid_tuning(agent_pid_stat, env, episodes, ep_length)
    rew_non  = run_pid_tuning(agent_pid_non,  env, episodes, ep_length)

    # 3) Plot learning curves
    plt.figure(figsize=(6,4))
    plt.plot(rew_stat, label='PID Stationary')
    plt.plot(rew_non,  label='PID NonStationary')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PID Controller Tuning with Bandit')
    plt.legend()
    plt.grid(True)

    # 4) Best arms
    best_stat = np.argmax(agent_pid_stat.Q)
    best_non  = np.argmax(agent_pid_non.Q)
    Kp_s, Ki_s, Kd_s = arms[best_stat]
    Kp_n, Ki_n, Kd_n = arms[best_non]

    print(f"Stationary best PID: Kp={Kp_s:.2f}, Ki={Ki_s:.2f}, Kd={Kd_s:.3f}")
    print(f"NonStat. best PID:    Kp={Kp_n:.2f}, Ki={Ki_n:.2f}, Kd={Kd_n:.3f}")

    # 5) Step response comparison
    T = 200
    r = np.ones(T)
    x_s = np.zeros(T)
    x_n = np.zeros(T)

    # Stationary PID response
    x = integral = e_prev = 0.0
    for t in range(T):
        e = x - r[t]
        integral += e*env.dt
        derivative = (e - e_prev)/env.dt if t>0 else 0.0
        u = -Kp_s*e - Ki_s*integral - Kd_s*derivative
        x = env.step(x, u)
        x_s[t] = x
        e_prev = e

    # NonStationary PID response
    x = integral = e_prev = 0.0
    for t in range(T):
        e = x - r[t]
        integral += e*env.dt
        derivative = (e - e_prev)/env.dt if t>0 else 0.0
        u = -Kp_n*e - Ki_n*integral - Kd_n*derivative
        x = env.step(x, u)
        x_n[t] = x
        e_prev = e

    plt.figure(figsize=(6,4))
    plt.plot(r, '--', label='Reference')
    plt.plot(x_s, label=f'Stationary PID')
    plt.plot(x_n, label=f'NonStat. PID')
    plt.xlabel('Time step')
    plt.ylabel('x(t)')
    plt.title('PID Controllers Step Response')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
