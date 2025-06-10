import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("q_learning_metrics.csv")


plt.figure(figsize=(10, 5))
plt.plot(df["Episode"], df["Avg_Latency"], label="Latency (ms)")
plt.plot(df["Episode"], df["Avg_Energy"], label="Energy (mW)")
plt.plot(df["Episode"], df["Avg_Throughput"], label="Throughput (Mbps)")
plt.xlabel("Episode")
plt.ylabel("Metric Value")
plt.title("Q-learning Performance over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
