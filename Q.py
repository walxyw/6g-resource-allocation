import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

class NetworkSimulator:
    def __init__(self, num_bs=3):
        self.num_bs = num_bs
        self.device_density = np.random.randint(50, 150, num_bs)
        self.channel_quality = np.random.uniform(0.7, 1.0, num_bs)
        self.load_history = []
        self.total_energy = 0.0

    def get_state(self):
        scaler = MinMaxScaler()
        if len(self.load_history) >= 3:
            normalized_load = scaler.fit_transform(np.array(self.load_history[-3:]).reshape(-1, 1)).flatten()
        else:
            normalized_load = [0.5] * 3
        
        
        state = [
            *[d / 150 for d in self.device_density],  
            *self.channel_quality,                   
            *normalized_load,                        
            self.total_energy / 1000 if self.total_energy > 0 else 0  
        ]
        return np.array(state)

    def discretize_state(self, state, bins=10):
       
        discretized = []
        for val in state:
            disc_val = np.digitize(val, np.linspace(0, 1, bins)) - 1
            discretized.append(disc_val)
        return tuple(discretized)

    def simulate(self, action):
     
        self.device_density = np.clip(
            self.device_density + np.random.randint(-15, 15, self.num_bs), 40, 160
        )
        self.channel_quality = np.clip(
            self.channel_quality + np.random.normal(0, 0.07, self.num_bs), 0.5, 1.0
        )

       
        base_latency = 0.05 + 0.1 * np.exp(-2 * np.mean(self.channel_quality))
        load_factor = np.sum([d/150 * a for d, a in zip(self.device_density, action)])
        latency = base_latency + 0.04 * load_factor**2
        
        energy = 0.7 + 0.15 * np.sum(action) + 0.08 * (np.mean(self.device_density)/100)
        throughput = 20 * np.sum(action) * np.mean(self.channel_quality)

        
        current_load = np.sum(action) / self.num_bs
        self.load_history.append(current_load)
        self.total_energy += energy

        return (
            max(0.03, latency + np.random.normal(0, 0.008)),
            energy,
            max(8, throughput + np.random.normal(0, 0.8)),
            self.total_energy
        )


class AdvancedQLearningAgent:
    def __init__(self, num_actions=5):
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        self.alpha = 0.2  
        self.gamma = 0.95  
        self.epsilon = 1.0
        self.min_epsilon = 0.05
        self.decay = 0.992
        
       
        self.base_actions = [
            [0.3, 0.4, 0.3],  
            [0.5, 0.3, 0.2],  
            [0.4, 0.4, 0.2],  
            [0.6, 0.2, 0.2],  
            [0.3, 0.3, 0.4]   
        ]

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.base_actions)
        else:
            action_idx = np.argmax(self.q_table[state])
            return self.base_actions[action_idx]

    def update(self, state, action, reward, next_state):
        action_idx = self.base_actions.index(action)
        current_q = self.q_table[state][action_idx]
        max_next_q = np.max(self.q_table[next_state])
        
    
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action_idx] = new_q


def advanced_train(episodes=800):
    simulator = NetworkSimulator()
    agent = AdvancedQLearningAgent()
    history = []
    
    for ep in range(episodes):
        state = simulator.get_state()
        disc_state = simulator.discretize_state(state)
        
        action = agent.get_action(disc_state)
        latency, energy, throughput, cumulative_energy = simulator.simulate(action)
        
        
        reward = (
            0.4 * np.tanh(throughput / 25)      
            - 0.4 * np.tanh(latency / 0.15)     
            - 0.15 * np.tanh(energy / 0.9)      
            + 0.05 * (1 - np.std(action))       
        )
        
        next_state = simulator.get_state()
        disc_next_state = simulator.discretize_state(next_state)
        agent.update(disc_state, action, reward, disc_next_state)
        
        history.append({
            'episode': ep,
            'latency': latency,
            'energy': energy,
            'cumulative_energy': cumulative_energy,
            'throughput': throughput,
            'action_std': np.std(action),
            'epsilon': agent.epsilon,
            'reward': reward
        })
        
        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.decay)
    
    return pd.DataFrame(history)


def enhanced_visualization(df):
    sns.set(style="whitegrid", palette="muted")
    plt.figure(figsize=(18, 15))
    
   
    plt.subplot(3, 2, 1)
    plt.plot(df['episode'], df['throughput'], 'g-', label='Throughput')
    plt.plot(df['episode'], df['latency']*100, 'r-', label='Latency (x100)')
    plt.xlabel('Episode')
    plt.ylabel('Performance')
    plt.title('Throughput and Latency Trend')
    plt.legend()
    plt.grid(True)
    
   
    plt.subplot(3, 2, 2)
    plt.plot(df['episode'], df['cumulative_energy'], 'b-', linewidth=2)
    plt.fill_between(df['episode'], df['cumulative_energy'], alpha=0.2)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Energy (kWh)')
    plt.title('Total Energy Consumption')
    plt.grid(True)
    
    
    plt.subplot(3, 2, 3)
    ax1 = plt.gca()
    ax1.plot(df['episode'], df['reward'], 'c-', label='Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color='c')
    ax1.tick_params(axis='y', labelcolor='c')
    
    ax2 = ax1.twinx()
    ax2.plot(df['episode'], df['epsilon'], 'm--', label='Epsilon')
    ax2.set_ylabel('Exploration Rate', color='m')
    ax2.tick_params(axis='y', labelcolor='m')
    plt.title('Reward and Exploration Rate')
    plt.grid(True)
    
   
    plt.subplot(3, 2, 4)
    plt.scatter(df['episode'], df['action_std'], c=df['reward'], cmap='viridis', alpha=0.6)
    plt.colorbar(label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Action Standard Deviation')
    plt.title('Resource Allocation Strategy')
    plt.grid(True)
    
   
    plt.subplot(3, 2, 5)
    sns.kdeplot(
        x=df['energy'], 
        y=df['throughput']/df['energy'], 
        fill=True, 
        cmap="Blues", 
        thresh=0.1
    )
    plt.xlabel('Energy Consumption (kW)')
    plt.ylabel('Throughput per Energy Unit')
    plt.title('Energy Efficiency Analysis')
    
   
    plt.subplot(3, 2, 6)
    sns.violinplot(y=df['latency'], inner="quartile", palette="Set3")
    plt.ylabel('Latency (ms)')
    plt.title('Latency Distribution')
    
    plt.tight_layout()
    plt.savefig('6g_resource_allocation_results.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    df = advanced_train(episodes=1000)
    enhanced_visualization(df)
    print("Training completed. Results saved to '6g_resource_allocation_results.png'")
