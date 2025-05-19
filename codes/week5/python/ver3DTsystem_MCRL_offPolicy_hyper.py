import numpy as np
import itertools
import matplotlib.pyplot as plt
from typing import List, Tuple
from collections import deque

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
        x0 = 0.0
        self.prev_u = 0.0
        e0 = x0 - self.ref_seq[0]
        s0 = self.discretize_error(e0)
        return x0, s0, 0


# --- Off-Policy Monte Carlo Control with Weighted Importance Sampling & ε-decay ---
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
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

        # Q(s,a) table and cumulative raw weights C(s,a)
        self.Q = np.zeros((num_states, num_actions))
        self.C = np.zeros((num_states, num_actions))

        # target policy: greedy w.r.t Q
        self.policy = np.zeros(num_states, dtype=int)

    def behavior_prob(self, s: int, a: int) -> float:
        greedy_a = self.policy[s]
        if a == greedy_a:
            return 1 - self.epsilon + self.epsilon / self.num_actions
        return self.epsilon / self.num_actions

    def select_action(self, s: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(self.policy[s])

    def update_episode(self, episode: List[Tuple[int, int, float]]):
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


# helper: sample a continuous error value given a discrete bin index
def sample_error_from_bin(s_idx: int, bins: np.ndarray) -> float:
    # treat bin 0 and bin N specially, else midpoint
    if s_idx == 0:
        return bins[0] - (bins[1] - bins[0]) / 2
    elif s_idx == len(bins):
        return bins[-1] + (bins[-1] - bins[-2]) / 2
    else:
        return 0.5 * (bins[s_idx - 1] + bins[s_idx])


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
    episodes: int,
    window_size: int = 10
) -> List[float]:
    """
    Sliding-window offline MC with Exploring-Starts:
      - Each episode starts at a random error-state and random first action
      - We maintain a deque of the last `window_size` transitions
      - After each new step, once the deque is full, we do an offline MC update
      - ε-decay happens once per episode
    """
    T = len(env.ref_seq)
    reward_history = []

    for ep in range(episodes):
        # -- Exploring-Starts --
        # pick a random discrete error state, then sample x accordingly
        s = np.random.randint(env.num_states)
        r0 = env.ref_seq[0]
        e0 = sample_error_from_bin(s, env.error_bins)
        x = e0

        # pick a random first action and set prev_u
        a = np.random.randint(env.num_actions)
        env.prev_u = env.actions[a]
        t = 0

        ep_reward = 0.0
        window = deque(maxlen=window_size)

        # run one episode with sliding window updates
        for _ in range(T):
            x_next, r, s_next, _ = env.step(x, t, a)
            window.append((s, a, r))
            ep_reward += r

            # once we have window_size samples, update MC
            if len(window) == window_size:
                agent.update_episode(list(window))

            # move to next step
            x, s, a = x_next, s_next, agent.select_action(s_next)
            t += 1

        # ε-decay once per episode
        agent.epsilon = max(agent.epsilon * agent.decay, agent.min_epsilon)
        reward_history.append(ep_reward)

    return reward_history


def simulate_policy(
    env: FirstOrderPlant,
    agent: OffPolicyMC
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run greedy policy to produce time, response, and control histories."""
    T = len(env.ref_seq)
    x, s, t = env.reset()
    x_hist = np.zeros(T)
    u_hist = np.zeros(T)

    for _ in range(T):
        a = agent.policy[s]
        x, _, s, u = env.step(x, t, a)
        x_hist[t] = x
        u_hist[t] = u
        t += 1

    return np.arange(T) * env.dt, x_hist, u_hist


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
    axs[1].plot(time, x_hist,     label="Response")
    axs[1].set(title="Reference vs Response", xlabel="Time (s)", ylabel="x(t)")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].step(time, u_hist, where="post")
    axs[2].set(title="Control Signal", xlabel="Time (s)", ylabel="u(t)")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

def evaluate_params(params, time, ref_seq, error_bins, episodes=200):
    """
    Verilen parametrelerle agent'ı eğit, policy'yi simüle et,
    ve referans-yanıt MSE'sini döndür.
    """
    # 1) ortam ve ajanı oluştur
    env = FirstOrderPlant(
        dt=params['dt'], ref_seq=ref_seq, error_bins=error_bins,
        error_weight=params['error_weight'],
        lam=params['lam'],
        beta=params['beta']
    )
    agent = OffPolicyMC(
        num_states=env.num_states,
        num_actions=env.num_actions,
        gamma=params['gamma'],
        epsilon=params['epsilon'],
        min_epsilon=0.01,
        decay=params['decay']
    )

    # 2) eğit
    _ = train_agent(env, agent, episodes=episodes, window_size=params['window_size'])

    # 3) son policy ile simüle et
    t_out, x_hist, _ = simulate_policy(env, agent)

    # 4) hata metriğini hesapla: ortalama kare hata
    mse = np.mean((x_hist - ref_seq[:len(x_hist)])**2)
    return mse

if __name__ == "__main__":
    # sabit tanımlar
    dt = 0.01
    total_time = 20.0
    time = np.arange(0, total_time, dt)
    ref_seq = generate_composite_reference(time, seed=42)
    error_bins = np.linspace(-5, 5, 21)

    # hiper-parametre grid'i
    grid = {
        'dt':            [dt],           # sabit
        'decay':         [0.995],        # sabit
        'error_weight':  [1.0, 10.0, 100.0, 1000.0],
        'lam':           [0.001, 0.01, 0.1],
        'beta':          [10.0],
        'gamma':         [0.9, 0.99, 0.999],
        'epsilon':       [0.1, 0.5],
        'window_size':   [5, 10, 20]
    }

    # toplam konfigürasyon sayısı
    all_combinations = list(itertools.product(*grid.values()))
    total_configs = len(all_combinations)

    best = {'mse': np.inf}

    for idx, vals in enumerate(all_combinations, start=1):
        params = dict(zip(grid.keys(), vals))

        mse = evaluate_params(params, time, ref_seq, error_bins, episodes=20)

        if mse < best['mse']:
            best = {'mse': mse, **params}

        # kaçıncı konfigürasyonda olduğumuzu ve toplamı yazdırıyoruz
        print(f"[{idx}/{total_configs}] denendi {params}, MSE={mse:.4f}")

    print("\n=== EN İYİ PARAMETRELER ===")
    print(best)

    # isterseniz en iyi parametrelerle bir kez daha eğitip plot edebilirsiniz:
    env = FirstOrderPlant(
        dt=best['dt'], ref_seq=ref_seq, error_bins=error_bins,
        error_weight=best['error_weight'], lam=best['lam'], beta=best['beta']
    )
    agent = OffPolicyMC(
        num_states=env.num_states,
        num_actions=env.num_actions,
        gamma=best['gamma'],
        epsilon=best['epsilon'],
        min_epsilon=0.01,
        decay=best['decay']
    )
    rewards = train_agent(env, agent, episodes=100, window_size=best['window_size'])
    t_out, x_hist, u_hist = simulate_policy(env, agent)
    plot_results(t_out, ref_seq, x_hist, u_hist, rewards)