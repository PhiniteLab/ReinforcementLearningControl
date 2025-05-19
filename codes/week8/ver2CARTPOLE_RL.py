import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym
from itertools import product

class FastCartPoleControl:
    def __init__(self, alpha, gamma, epsilon, epsilon_decay=0.999, min_epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Aynı binler
        self.cart_pos_bins = np.arange(-2.4, 2.5, 0.2)
        self.cart_vel_bins = np.arange(-3.0, 3.1, 0.25)
        self.pole_angle_bins = np.arange(-0.21, 0.215, 0.02)
        self.pole_vel_bins = np.arange(-2.0, 2.1, 0.15)

        # NumPy array Q-table (daha hızlı!)
        self.q_table = np.zeros((
            len(self.cart_pos_bins)+1,
            len(self.cart_vel_bins)+1,
            len(self.pole_angle_bins)+1,
            len(self.pole_vel_bins)+1,
            2
        ))

    def _get_state_idx(self, state):
        cp = int(np.digitize([state[0]], self.cart_pos_bins)[0])
        cv = int(np.digitize([state[1]], self.cart_vel_bins)[0])
        pa = int(np.digitize([state[2]], self.pole_angle_bins)[0])
        pv = int(np.digitize([state[3]], self.pole_vel_bins)[0])
        return (cp, cv, pa, pv)

    def learn(self, env, num_episodes):
        self.epsilon = float(self.epsilon)
        for episode in range(num_episodes):
            obs = env.reset()
            state = obs[0] if isinstance(obs, tuple) else obs
            idx = self._get_state_idx(state)
            done = False
            while not done:
                if np.random.rand() < self.epsilon:
                    action = np.random.choice([0, 1])
                else:
                    action = int(np.argmax(self.q_table[idx]))
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result
                idx_next = self._get_state_idx(next_state)
                q = self.q_table[(*idx, action)]
                q_max_next = np.max(self.q_table[idx_next])
                self.q_table[(*idx, action)] += self.alpha * (reward + self.gamma * q_max_next - q)
                idx = idx_next
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        # Sadece seyrek print!
            if (episode+1) % 500 == 0 or episode == 0:
                print(f"Episode {episode+1}/{num_episodes}   Epsilon: {self.epsilon:.4f}")

    def test(self, env, num_episodes):
        rewards_list = []
        for episode in range(num_episodes):
            obs = env.reset()
            state = obs[0] if isinstance(obs, tuple) else obs
            idx = self._get_state_idx(state)
            done = False
            rewards = 0.0
            while not done:
                action = int(np.argmax(self.q_table[idx]))
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result
                idx = self._get_state_idx(next_state)
                rewards += reward
            rewards_list.append(rewards)
        mean_reward = np.mean(rewards_list)
        print(f"Test episodes: {rewards_list}  Mean: {mean_reward:.1f}")
        return mean_reward

def save_agent(agent, filename="best_qtable.npy"):
    np.save(filename, agent.q_table)
    print(f"Q-table başarıyla '{filename}' olarak kaydedildi.")

def load_agent(agent, filename="best_qtable.npy"):
    agent.q_table = np.load(filename)
    print(f"Q-table '{filename}' dosyasından yüklendi.")

def animate_cartpole(agent, env, max_steps=500, max_trials=10):
    for trial in range(max_trials):
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        states = [state]
        done = False
        steps = 0
        while not done and steps < max_steps:
            idx = agent._get_state_idx(state)
            action = int(np.argmax(agent.q_table[idx]))
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result
            state = next_state
            states.append(state)
            steps += 1
        if steps >= max_steps:
            print(f"Agent, {max_steps} adım boyunca devrilmedi! Animasyon başlatılıyor...")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.set_xlim(-2.4, 2.4)
            ax.set_ylim(-0.5, 1.2)
            cart_width = 0.4
            cart_height = 0.2
            pole_length = 1.0
            cart = plt.Rectangle((0, 0), cart_width, cart_height, color='black')
            pole, = ax.plot([], [], lw=4, color='blue')
            ax.add_patch(cart)
            def update(i):
                x, _, theta, _ = states[i]
                cart.set_xy((x - cart_width/2, 0))
                pole_x = x + pole_length * np.sin(theta)
                pole_y = cart_height + pole_length * np.cos(theta)
                pole.set_data([x, pole_x], [cart_height, pole_y])
                return cart, pole
            ani = animation.FuncAnimation(
                fig, update, frames=len(states), interval=30, blit=True, repeat=False
            )
            plt.title(f'CartPole - Q-Learning Agent, {max_steps} adım dengede')
            plt.show()
            return
        else:
            print(f"Deneme {trial+1}: Agent devrildi (adım sayısı: {steps})")
    print(f"FAIL: Agent hiç {max_steps} adım dengede kalamadı!")
    return

def hyperparameter_search():
    env = gym.make("CartPole-v1")
    alphas = [0.05, 0.1, 0.2]
    gammas = [0.95, 0.99, 0.999]
    epsilons = [0.5, 1.0]
    epsilon_decays = [0.99, 0.995, 0.999]
    min_epsilons = [0.01, 0.05]
    num_episodes = 2000
    num_test_episodes = 5

    results = []
    param_grid = list(product(alphas, gammas, epsilons, epsilon_decays, min_epsilons))
    print(f"{len(param_grid)} kombinasyon deneniyor...")

    for idx, (alpha, gamma, epsilon, epsilon_decay, min_epsilon) in enumerate(param_grid):
        agent = FastCartPoleControl(alpha, gamma, epsilon, epsilon_decay, min_epsilon)
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
    print(f"Ortalama test reward: {best['mean_reward']:.1f}")

    save_agent(best['agent'], "best_qtable.npy")
    print("\nEn iyi agent ile animasyon başlatılıyor...")
    animate_cartpole(best['agent'], env, max_steps=2000, max_trials=10)

if __name__ == "__main__":
    # Hiperparametre araması ve en iyi agent'ı kaydet
    hyperparameter_search()

    # Kayıtlı agent'ı yükle ve test/animasyon yap
    env = gym.make("CartPole-v1")
    env._max_episode_steps = 2000  # veya istediğin değer
    loaded_agent = FastCartPoleControl(alpha=0.1, gamma=0.99, epsilon=0.01)
    load_agent(loaded_agent, "best_qtable.npy")
    print("\nYüklenen agent ile test:")
    loaded_agent.test(env, num_episodes=3)
    print("\nYüklenen agent ile animasyon:")
    animate_cartpole(loaded_agent, env, max_steps=2000, max_trials=10)
