import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product

# --- Yeni aksiyonlar ---
force_values = np.arange(-10, 11, 1.0)   # -10 ... +10 (21 seviye)
num_actions = len(force_values)

# === MCK Modeli (Mass-Spring-Damper) Ortamı ===
class MassSpringDamperEnv:
    def __init__(self, m=1.0, k=1.0, c=0.2, dt=0.02, x_threshold=2.5, u_max=10.0, max_episode_steps=500, ref=1.0):
        self.m = m
        self.k = k
        self.c = c
        self.dt = dt
        self.x_threshold = x_threshold
        self.u_max = u_max
        self.max_episode_steps = max_episode_steps
        self.ref = ref  # Referans konum
        self.state = None
        self.step_count = 0
        self.last_force = 0.0  # <-- Son uygulanan kontrol (ilk adım için)

        # Bu katsayıları ihtiyacına göre değiştirebilirsin
        self.lam_u = 0.02      # u^2 ceza katsayısı
        self.lam_du = 0.1      # delta_u^2 ceza katsayısı

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.uniform(low=[-0.05, -0.05], high=[0.05, 0.05])
        self.step_count = 0
        self.last_force = 0.0  # Sıfırdan başla
        return self.state.copy()

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action)
        force = force_values[action]

        # u sınırı (gerekirse)
        force = np.clip(force, -self.u_max, self.u_max)

        x, x_dot = self.state
        x_ddot = (force - self.c * x_dot - self.k * x) / self.m
        x_dot = x_dot + self.dt * x_ddot
        x = x + self.dt * x_dot

        self.state = np.array([x, x_dot])
        self.step_count += 1

        done = bool(
            abs(x) > self.x_threshold or self.step_count >= self.max_episode_steps
        )

        # --- Reward tanımı ---
        error = x - self.ref
        u = force
        delta_u = u - self.last_force  # Değişim

        # Ceza fonksiyonu (yumuşak, L2 önerilir!)
        reward = -10.0*abs(error) - self.lam_u * (u ** 2) - self.lam_du * (delta_u ** 2)

        # Büyük sapmalara ekstra ceza
        if done and abs(x) > self.x_threshold:
            reward -= 10

        self.last_force = u  # Gelecek adımda delta_u için

        info = {"ref": self.ref, "u": u, "delta_u": delta_u}
        return self.state.copy(), reward, done, info


# === Q-Learning Ajanı (MCK için düzenlenmiş) ===
class FastMCKControl:
    def __init__(self, alpha, gamma, epsilon, epsilon_decay=0.999, min_epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.x_bins = np.linspace(-2.5, 2.5, 30)
        self.x_dot_bins = np.linspace(-3.0, 3.0, 30)

        self.q_table = np.zeros((
            len(self.x_bins)+1,
            len(self.x_dot_bins)+1,
            num_actions
        ))

    def _get_state_idx(self, state):
        x, x_dot = state
        xb = int(np.digitize([x], self.x_bins)[0])
        vb = int(np.digitize([x_dot], self.x_dot_bins)[0])
        return (xb, vb)

    def learn(self, env, num_episodes):
        self.epsilon = float(self.epsilon)
        for episode in range(num_episodes):
            state = env.reset()
            idx = self._get_state_idx(state)
            done = False
            while not done:
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(num_actions)   # Güncellendi!
                else:
                    action = int(np.argmax(self.q_table[idx]))
                next_state, reward, done, _ = env.step(action)
                idx_next = self._get_state_idx(next_state)
                q = self.q_table[(*idx, action)]
                q_max_next = np.max(self.q_table[idx_next])
                self.q_table[(*idx, action)] += self.alpha * (reward + self.gamma * q_max_next - q)
                idx = idx_next
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            if (episode+1) % 500 == 0 or episode == 0:
                print(f"Episode {episode+1}/{num_episodes}   Epsilon: {self.epsilon:.4f}")

    def test(self, env, num_episodes):
        rewards_list = []
        for episode in range(num_episodes):
            state = env.reset()
            idx = self._get_state_idx(state)
            done = False
            rewards = 0.0
            while not done:
                action = int(np.argmax(self.q_table[idx]))
                next_state, reward, done, _ = env.step(action)
                idx = self._get_state_idx(next_state)
                rewards += reward
            rewards_list.append(rewards)
        mean_reward = np.mean(rewards_list)
        print(f"Test episodes: {rewards_list}  Mean: {mean_reward:.2f}")
        return mean_reward

def save_agent(agent, filename="best_mck_qtable.npy"):
    np.save(filename, agent.q_table)
    print(f"Q-table başarıyla '{filename}' olarak kaydedildi.")

def load_agent(agent, filename="best_mck_qtable.npy"):
    agent.q_table = np.load(filename)
    print(f"Q-table '{filename}' dosyasından yüklendi.")

def animate_mck(agent, env, max_steps=500, max_trials=3):
    for trial in range(max_trials):
        state = env.reset()
        states = [state.copy()]
        actions = []
        forces = []
        done = False
        steps = 0
        while not done and steps < max_steps:
            idx = agent._get_state_idx(state)
            action = int(np.argmax(agent.q_table[idx]))
            force = force_values[action]   # Güncellendi!
            actions.append(action)
            forces.append(force)
            next_state, reward, done, info = env.step(action)
            state = next_state
            states.append(state.copy())
            steps += 1
        print(f"Deneme {trial+1}: {steps} adım (final x = {state[0]:.2f})")

        x_data = [s[0] for s in states]
        ref = info.get("ref", 0.0)
        t_data = np.arange(len(x_data)) * env.dt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(hspace=0.33)
        ax1.set_xlim(-env.x_threshold-0.2, env.x_threshold+0.2)
        ax1.set_ylim(-0.6, 0.6)
        ax1.axvline(ref, color='red', linestyle='--', label='Referans')
        mass, = ax1.plot([], [], 'bo', markersize=18, label='Kütle (m)')
        spring_line, = ax1.plot([], [], 'g-', lw=3)
        ax1.legend()
        ax1.set_title(f'MCK RL - Deneme {trial+1} (Pozisyon)')
        ax2.set_xlim(0, len(x_data)*env.dt)
        ax2.set_ylim(-env.u_max-1, env.u_max+1)
        force_line, = ax2.plot([], [], 'r-', lw=2, label='Uygulanan Kuvvet (N)')
        ax2.set_xlabel('Zaman (s)')
        ax2.set_ylabel('Kuvvet')
        ax2.legend()
        ax2.axhline(0, color='k', linestyle='--', lw=0.8)

        def update(i):
            x = x_data[i]
            spring_x = np.linspace(0, x, 100)
            spring_y = np.zeros_like(spring_x)
            mass.set_data([x], [0])
            spring_line.set_data(spring_x, spring_y)
            if i > 0:
                force_line.set_data(t_data[:i], forces[:i])
            else:
                force_line.set_data([], [])
            return mass, spring_line, force_line

        ani = animation.FuncAnimation(
            fig, update, frames=len(x_data), interval=30, blit=True, repeat=False
        )
        plt.show()
        if abs(state[0] - ref) < 0.05:
            print("Ajan referansa başarıyla ulaştı.")
        else:
            print("Ajan referansa yeterince yakın değil.")

def hyperparameter_search():
    env = MassSpringDamperEnv(max_episode_steps=300, ref=1.0)
    alphas = [0.05, 0.1, 0.15, 0.18, 0.21, 0.25]
    gammas = [0.95, 0.98, 0.99, 0.995, 0.999]
    epsilons = [1.0, 0.7]
    epsilon_decays = [0.995, 0.998, 0.999]
    min_epsilons = [0.01, 0.05]
    num_episodes = 500
    num_test_episodes = 3

    results = []
    param_grid = list(product(alphas, gammas, epsilons, epsilon_decays, min_epsilons))
    print(f"{len(param_grid)} kombinasyon deneniyor...")

    for idx, (alpha, gamma, epsilon, epsilon_decay, min_epsilon) in enumerate(param_grid):
        agent = FastMCKControl(alpha, gamma, epsilon, epsilon_decay, min_epsilon)
        print(f"\n[{idx+1}/{len(param_grid)}] alpha={alpha}, gamma={gamma}, epsilon={epsilon}, "
              f"epsilon_decay={epsilon_decay}, min_epsilon={min_epsilon}")
        agent.learn(env, num_episodes=num_episodes)
        mean_reward = agent.test(env, num_test_episodes)
        results.append({
            'params': (alpha, gamma, epsilon, epsilon_decay, min_epsilon),
            'mean_reward': mean_reward,
            'agent': agent
        })

    best = max(results, key=lambda x: x['mean_reward'])
    print("\n=== EN İYİ PARAMETRELER ===")
    print(f"alpha={best['params'][0]}, gamma={best['params'][1]}, epsilon={best['params'][2]}, "
          f"epsilon_decay={best['params'][3]}, min_epsilon={best['params'][4]}")
    print(f"Ortalama test reward: {best['mean_reward']:.2f}")

    save_agent(best['agent'], "best_mck_qtable.npy")
    print("\nEn iyi agent ile animasyon başlatılıyor...")
    animate_mck(best['agent'], env, max_steps=300, max_trials=2)

if __name__ == "__main__":
    hyperparameter_search()

    env = MassSpringDamperEnv(max_episode_steps=300, ref=1.0)
    loaded_agent = FastMCKControl(alpha=0.15, gamma=0.999, epsilon=0.01)
    load_agent(loaded_agent, "best_mck_qtable.npy")
    print("\nYüklenen agent ile test:")
    loaded_agent.test(env, num_episodes=2)
    print("\nYüklenen agent ile animasyon:")
    animate_mck(loaded_agent, env, max_steps=300, max_trials=1)
