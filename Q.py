import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from enum import Enum

class DeviceType(Enum):
    SENSOR = 1      
    ACTUATOR = 2
    APPLIANCE = 3   

class IoTDevice:
    def __init__(self, device_type):
        self.type = device_type
        
        
        if device_type == DeviceType.SENSOR:
            self.packet_rate = 50      
            self.packet_size = 128      
            self.power_factor = 0.7
        elif device_type == DeviceType.ACTUATOR:
            self.packet_rate = 200     
            self.packet_size = 256      
            self.power_factor = 1.0
        elif device_type == DeviceType.APPLIANCE:
            self.packet_rate = 100     
            self.packet_size = 512      
            self.power_factor = 0.8

class NetworkSimulator:
    def __init__(self, num_bs=3, num_devices=5000):
        self.num_bs = num_bs
        self.num_devices = num_devices
        
        
        self.devices = []
        for _ in range(int(num_devices * 0.4)):
            self.devices.append(IoTDevice(DeviceType.SENSOR))
        for _ in range(int(num_devices * 0.35)):
            self.devices.append(IoTDevice(DeviceType.ACTUATOR))
        for _ in range(int(num_devices * 0.25)):
            self.devices.append(IoTDevice(DeviceType.APPLIANCE))
        
        
        self.bs_devices = [[] for _ in range(num_bs)]
        for device in self.devices:
            bs_idx = random.randint(0, num_bs-1)
            self.bs_devices[bs_idx].append(device)
        
        
        self.device_density = [len(devs) for devs in self.bs_devices]
        self.channel_quality = np.random.uniform(0.7, 1.0, num_bs)
        self.load_history = [[] for _ in range(num_bs)]
        self.total_energy = 0.0
        self.time_step = 0

    def get_state(self):
        
        state = []
        
        
        max_density = max(self.device_density) if max(self.device_density) > 0 else 1
        state.extend([d / max_density for d in self.device_density])
        
        
        state.extend(self.channel_quality)
        
        
        for i in range(self.num_bs):
            if len(self.load_history[i]) >= 3:
                
                ema = self.load_history[i][-1] * 0.5 + self.load_history[i][-2] * 0.3 + self.load_history[i][-3] * 0.2
                state.append(ema)
            else:
                state.append(0.5)  
        
        
        energy_norm = min(1.0, self.total_energy / 1000)  
        state.append(energy_norm)
        
        return np.array(state)

    def discretize_state(self, state, bins=10):
        
        discretized = []
        for val in state:
            disc_val = np.digitize(val, np.linspace(0, 1, bins)) - 1
            discretized.append(disc_val)
        return tuple(discretized)

    def simulate(self, action):
        
        self.time_step += 1
        
        
        self._update_network_state()
        
        
        bs_load = [0.0] * self.num_bs
        for i in range(self.num_bs):
            for device in self.bs_devices[i]:
                
                burst_factor = 1.0
                if random.random() < 0.1:  
                    burst_factor = random.uniform(1.5, 3.0)
                
            
                bs_load[i] += device.packet_rate * burst_factor * device.packet_size / 1e6  
        
        
        for i in range(self.num_bs):
            if len(self.load_history[i]) >= 5:
                self.load_history[i].pop(0)
            self.load_history[i].append(bs_load[i] / max(bs_load) if max(bs_load) > 0 else 0)
        
        
        latency, energy, throughput = self._calculate_performance(action, bs_load)
        
        
        self.total_energy += energy
        
        return latency, energy, throughput

    def _update_network_state(self):
        
        
        for i in range(self.num_bs):
            change = random.randint(-5, 5)
            if 0 <= self.device_density[i] + change <= 160:
                self.device_density[i] += change
        
        
        self.channel_quality = np.clip(
            self.channel_quality + np.random.normal(0, 0.05, self.num_bs), 0.5, 1.0
        )
        
        
        if self.time_step % 100 == 0:
            hotspot_bs = random.randint(0, self.num_bs-1)
            self.device_density[hotspot_bs] = min(160, self.device_density[hotspot_bs] + 30)

    def _calculate_performance(self, action, bs_load):
        
        
        base_latency = 0.05 + 0.1 * np.exp(-2 * np.mean(self.channel_quality))
        load_factor = np.sum([l * a for l, a in zip(bs_load, action)]) / np.sum(bs_load)
        latency = base_latency + 0.04 * load_factor**2
        
        
        energy = 0.0
        for i in range(self.num_bs):
            for device in self.bs_devices[i]:
                
                device_energy = 0.1 + 0.05 * device.power_factor * action[i]
                energy += device_energy
        
        
        throughput = 0.0
        for i in range(self.num_bs):
            sector_throughput = bs_load[i] * self.channel_quality[i] * action[i]
            throughput += sector_throughput
        
        
        latency = max(0.03, latency + np.random.normal(0, 0.005))
        energy = max(0.1, energy + np.random.normal(0, 0.05))
        throughput = max(5, throughput + np.random.normal(0, 0.5))
        
        return latency, energy, throughput


class BaseAllocationStrategy:
    
    def get_action(self, state):
        raise NotImplementedError("Subclasses must implement get_action")
    
    @staticmethod
    def get_base_actions(num_bs):
        
        if num_bs == 3:
            return [
                [0.3, 0.4, 0.3],  
                [0.5, 0.3, 0.2],  
                [0.4, 0.4, 0.2],  
                [0.6, 0.2, 0.2],  
                [0.3, 0.3, 0.4]   
            ]
        else:
            
            base_action = [1.0/num_bs] * num_bs
            actions = [base_action]
            for i in range(num_bs):
                action = base_action.copy()
                action[i] = min(0.6, action[i] + 0.3)  
                for j in range(num_bs):
                    if j != i:
                        action[j] = action[j] * (1 - 0.3/(num_bs-1))
                actions.append(action)
            return actions


class StaticAllocation(BaseAllocationStrategy):
    
    def __init__(self, num_bs):
        self.base_actions = self.get_base_actions(num_bs)
        self.action = self.base_actions[0]  
    
    def get_action(self, state):
        return self.action


class HeuristicAllocation(BaseAllocationStrategy):
    
    def __init__(self, num_bs):
        self.base_actions = self.get_base_actions(num_bs)
        self.num_bs = num_bs
    
    def get_action(self, state):
        
        avg_load = sum(state[:self.num_bs]) / self.num_bs
        
        if avg_load < 0.4:
            return self.base_actions[4]  
        elif avg_load > 0.75:
            return self.base_actions[1]  
        else:
            return self.base_actions[0]  


class QLearningAgent(BaseAllocationStrategy):
    
    def __init__(self, num_bs, num_actions=5):
        super().__init__()
        self.base_actions = self.get_base_actions(num_bs)
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        self.alpha = 0.2  
        self.gamma = 0.95  
        self.epsilon = 1.0  
        self.min_epsilon = 0.05
        self.decay = 0.995  

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
        
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)


def calculate_reward(latency, throughput, energy, action):
    
    
    R_thru = min(1.0, throughput / 30.0)  
    
    
    R_lat = max(0.0, 1.0 - (latency / 0.2))  
    
    
    R_energy = max(0.0, 1.0 - (energy / 1.0))  
    
    
    R_fair = 1.0 - np.std(action)  
    
    
    reward = (
        0.40 * R_thru +
        0.40 * R_lat +
        0.15 * R_energy +
        0.05 * R_fair
    )
    
    return reward


def train_evaluate(episodes=1000, num_bs=3, num_devices=5000):
    
    
    simulator = NetworkSimulator(num_bs=num_bs, num_devices=num_devices)
    q_agent = QLearningAgent(num_bs)
    static_agent = StaticAllocation(num_bs)
    heuristic_agent = HeuristicAllocation(num_bs)
    
    
    history = []
    
    for ep in range(episodes):
        
        state = simulator.get_state()
        disc_state = simulator.discretize_state(state)
        
        
        strategies = [
            ('Q-learning', q_agent),
            ('Static', static_agent),
            ('Heuristic', heuristic_agent)
        ]
        
        ep_results = {'episode': ep}
        
        for strategy_name, agent in strategies:
            
            action = agent.get_action(disc_state)
            
            
            latency, energy, throughput = simulator.simulate(action)
            
            
            reward = calculate_reward(latency, throughput, energy, action)
            
            
            if strategy_name == 'Q-learning':
                next_state = simulator.get_state()
                disc_next_state = simulator.discretize_state(next_state)
                q_agent.update(disc_state, action, reward, disc_next_state)
            
            
            ep_results[f'{strategy_name}_latency'] = latency
            ep_results[f'{strategy_name}_energy'] = energy
            ep_results[f'{strategy_name}_throughput'] = throughput
            ep_results[f'{strategy_name}_reward'] = reward
        
        
        ep_results['epsilon'] = q_agent.epsilon
        history.append(ep_results)
        
        
        if ep % 100 == 0:
            print(f"Episode {ep}/{episodes} completed")
    
    return pd.DataFrame(history)


def enhanced_visualization(df):
    
    sns.set(style="whitegrid", palette="muted")
    plt.figure(figsize=(18, 15))
    
    
    plt.subplot(3, 2, 1)
    plt.plot(df['episode'], df['Q-learning_latency'], 'b-', label='Q-learning')
    plt.plot(df['episode'], df['Static_latency'], 'r-', label='Static')
    plt.plot(df['episode'], df['Heuristic_latency'], 'g-', label='Heuristic')
    plt.xlabel('Episode')
    plt.ylabel('Latency (ms)')
    plt.title('Latency Comparison')
    plt.legend()
    plt.grid(True)
    
    
    plt.subplot(3, 2, 2)
    plt.plot(df['episode'], df['Q-learning_throughput'], 'b-', label='Q-learning')
    plt.plot(df['episode'], df['Static_throughput'], 'r-', label='Static')
    plt.plot(df['episode'], df['Heuristic_throughput'], 'g-', label='Heuristic')
    plt.xlabel('Episode')
    plt.ylabel('Throughput (Mbps)')
    plt.title('Throughput Comparison')
    plt.legend()
    plt.grid(True)
    
    
    plt.subplot(3, 2, 3)
    plt.plot(df['episode'], df['Q-learning_energy'], 'b-', label='Q-learning')
    plt.plot(df['episode'], df['Static_energy'], 'r-', label='Static')
    plt.plot(df['episode'], df['Heuristic_energy'], 'g-', label='Heuristic')
    plt.xlabel('Episode')
    plt.ylabel('Energy Consumption (kW)')
    plt.title('Energy Consumption Comparison')
    plt.legend()
    plt.grid(True)
    
    
    plt.subplot(3, 2, 4)
    plt.plot(df['episode'], df['Q-learning_reward'], 'b-', label='Q-learning')
    plt.plot(df['episode'], df['Static_reward'], 'r-', label='Static')
    plt.plot(df['episode'], df['Heuristic_reward'], 'g-', label='Heuristic')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Comparison')
    plt.legend()
    plt.grid(True)
    
    
    plt.subplot(3, 2, 5)
    plt.plot(df['episode'], df['epsilon'], 'm-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Exploration Rate (ε)')
    plt.title('Q-learning Exploration Rate Decay')
    plt.grid(True)
    
    
    plt.subplot(3, 2, 6)
    
    last_100 = df.tail(100)
    latency_data = [
        last_100['Q-learning_latency'],
        last_100['Static_latency'],
        last_100['Heuristic_latency']
    ]
    
    plt.boxplot(latency_data, labels=['Q-learning', 'Static', 'Heuristic'])
    plt.ylabel('Latency (ms)')
    plt.title('Final Performance Comparison (last 100 episodes)')
    
    plt.tight_layout()
    plt.savefig('6g_resource_allocation_comparison.png', dpi=300)
    plt.show()


def generate_performance_table(df):

    
    last_100 = df.tail(100)
    
    
    performance = {
        'Strategy': ['Q-learning', 'Static', 'Heuristic'],
        'Avg. Latency (ms)': [
            last_100['Q-learning_latency'].mean(),
            last_100['Static_latency'].mean(),
            last_100['Heuristic_latency'].mean()
        ],
        'Avg. Throughput (Mbps)': [
            last_100['Q-learning_throughput'].mean(),
            last_100['Static_throughput'].mean(),
            last_100['Heuristic_throughput'].mean()
        ],
        'Avg. Energy (kW)': [
            last_100['Q-learning_energy'].mean(),
            last_100['Static_energy'].mean(),
            last_100['Heuristic_energy'].mean()
        ],
        'Avg. Reward': [
            last_100['Q-learning_reward'].mean(),
            last_100['Static_reward'].mean(),
            last_100['Heuristic_reward'].mean()
        ]
    }
    
    
    heuristic_latency = last_100['Heuristic_latency'].mean()
    heuristic_throughput = last_100['Heuristic_throughput'].mean()
    heuristic_energy = last_100['Heuristic_energy'].mean()
    
    performance['Latency Improvement'] = [
        f"{((heuristic_latency - performance['Avg. Latency (ms)'][0]) / heuristic_latency * 100):.1f}%",
        "-",
        "-"
    ]
    
    performance['Throughput Improvement'] = [
        f"{((performance['Avg. Throughput (Mbps)'][0] - heuristic_throughput) / heuristic_throughput * 100):.1f}%",
        "-",
        "-"
    ]
    
    performance['Energy Improvement'] = [
        f"{((heuristic_energy - performance['Avg. Energy (kW)'][0]) / heuristic_energy * 100):.1f}%",
        "-",
        "-"
    ]
    
    
    perf_df = pd.DataFrame(performance)
    
    
    print("\nPerformance Summary Table (last 100 episodes):")
    print(perf_df.to_string(index=False))
    
    
    perf_df.to_csv('performance_summary.csv', index=False)
    print("Performance summary saved to 'performance_summary.csv'")


if __name__ == '__main__':
    
    results_df = train_evaluate(episodes=1000, num_bs=3, num_devices=5000)
    
    
    enhanced_visualization(results_df)
    
    
    generate_performance_table(results_df)
    
    
    results_df.to_csv('training_results.csv', index=False)
    print("Training completed. Results saved to CSV files.")
