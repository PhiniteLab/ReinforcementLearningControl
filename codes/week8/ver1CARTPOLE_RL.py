import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym
import pickle
from itertools import product

# ------------------- AGENT CLASS -------------------

class CartPoleControl:
    """
    Cart Pole control using Q-Learning.
    Parameters:
    alpha (float): Learning rate
    gamma (float): Discount factor
    epsilon (float): Exploration rate
    """
    def __init__(self, alpha, gamma, epsilon, epsilon_decay=0.999, min_epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}
        # Discretization bins
        self.cart_pos_bins = np.arange(-2.4, 2.5, 0.2)
        self.cart_vel_bins = np.arange(-3.0, 3.1, 0.25)
        self.pole_angle_bins = np.arange(-0.21, 0.215, 0.02)
        self.pole_vel_bins = np.arange(-2.0, 2.1, 0.15)

    def _discretize(self, value, bins):
        return int(np.digitize([value], bins)[0])

    def _get_state(self, state):
        cart_position, cart_velocity, pole_angle, pole_velocity = state
        cp = self._discretize(cart_position, self.cart_pos_bins)
        cv = self._discretize(cart_velocity, self.cart_vel_bins)
        pa = self._discretize(pole_angle, self.pole_angle_bins)
        pv = self._discretize(pole_velocity, self.pole_vel_bins)
        return (cp, cv, pa, pv)

    def _get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def _set_q_value(self, state, action, value):
        self.q_table[(state, action)] = value

    def _choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        else:
            q_values = [self._get_q_value(state, action) for action in [0, 1]]
            return int(np.argmax(q_values))

    def learn(self, env, num_episodes):
        self.epsilon = float(self.epsilon)
        for episode in range(num_episodes):
            obs = env.reset()
            state = obs[0] if isinstance(obs, tuple) else obs
            state = self._get_state(state)
            done = False
            while not done:
                action = self._choose_action(state)
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result
                next_state_disc = self._get_state(next_state)
                q_value = self._get_q_value(state, action)
                next_q_value = max([self._get_q_value(next_state_disc, a) for a in [0, 1]])
                q_value += self.alpha * (reward + self.gamma * next_q_value - q_value)
                self._set_q_value(state, action, q_value)
                state = next_state_disc
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def test(self, env, num_episodes):
        rewards_list = []
        for episode in range(num_episodes):
            obs = env.reset()
            state = obs[0] if isinstance(obs, tuple) else obs
            state = self._get_state(state)
            done = False
            rewards = 0.0
            while not done:
                action = int(np.argmax([self._get_q_value(state, a) for a in [0, 1]]))
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result
                state = self._get_state(next_state)
                rewards += reward
            rewards_list.append(rewards)
        mean_reward = np.mean(rewards_list)
        print(f"Test episodes: {rewards_list}  Mean: {mean_reward:.1f}")
        return mean_reward

# ------------------- Q-TABLE SAVE/LOAD -------------------

def save_agent(agent, filename="best_qtable.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(agent.q_table, f)
    print(f"Q-table başarıyla '{filename}' olarak kaydedildi.")

def load_agent(agent, filename="best_qtable.pkl"):
    with open(filename, "rb") as f:
        agent.q_table = pickle.load(f)
    print(f"Q-table '{filename}' dosyasından yüklendi.")

# ------------------- ANİMASYON FONKSİYONU -------------------

def animate_cartpole(agent, env, max_steps=500, max_trials=10):
    for trial in range(max_trials):
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        states = [state]
        done = False
        steps = 0
        while not done and steps < max_steps:
            disc_state = agent._get_state(state)
            action = int(np.argmax([agent._get_q_value(disc_state, a) for a in [0, 1]]))
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

# ------------------- HİPERPARAMETRE ARAMA (OPSİYONEL) -------------------

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
        agent = CartPoleControl(alpha, gamma, epsilon, epsilon_decay, min_epsilon)
        print(f"\n[{idx+1}/{len(param_grid)}] alpha={alpha}, gamma={gamma}, epsilon={epsilon}, "
              f"epsilon_decay={epsilon_decay}, min_epsilon={min_epsilon}")
        agent.learn(env, num_episodes=num_episodes)
        mean_reward = agent.test(env, num_test_episodes)
        results.append({
            'params': (alpha, gamma, epsilon, epsilon_decay, min_epsilon),
            'mean_reward': mean_reward,
            'agent': agent
        })

    # En iyi sonucu bul
    best = max(results, key=lambda x: x['mean_reward'])
    print("\n=== EN İYİ PARAMETRELER ===")
    print(f"alpha={best['params'][0]}, gamma={best['params'][1]}, epsilon={best['params'][2]}, "
          f"epsilon_decay={best['params'][3]}, min_epsilon={best['params'][4]}")
    print(f"Ortalama test reward: {best['mean_reward']:.1f}")

    # En iyi ajanı kaydet
    save_agent(best['agent'], "best_qtable.pkl")

    # En iyi agent ile animasyon
    print("\nEn iyi agent ile animasyon başlatılıyor...")
    animate_cartpole(best['agent'], env, max_steps=500, max_trials=10)

# ------------------- KULLANIM AKIŞI -------------------

if __name__ == "__main__":
    # --- Hiperparametre aramasıyla eğitim ve kaydetme (en iyi ajanı kaydeder) ---
    hyperparameter_search()

    # --- Aşağıda kaydedilmiş bir ajanı yükleyip tekrar oynat ---
    env = gym.make("CartPole-v1")
    loaded_agent = CartPoleControl(alpha=0.1, gamma=0.99, epsilon=0.01)  # alpha/gamma burada önemli değil
    load_agent(loaded_agent, "best_qtable.pkl")
    print("\nYüklenen agent ile test:")
    loaded_agent.test(env, num_episodes=3)
    print("\nYüklenen agent ile animasyon:")
    animate_cartpole(loaded_agent, env, max_steps=500, max_trials=10)
