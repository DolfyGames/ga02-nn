import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load only the available model log
df_base = pd.read_csv('model_logs/v17.1.csv')

# Calculate moving averages
df_base['length_mean_ma'] = df_base['length_mean'].rolling(10).mean()
df_base['loss_ma'] = df_base['loss'].rolling(10).mean()

# Plot 1: Training Progress - Mean Length
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

axs[0].set_title('v17.1 Snake Mean Length Over Training')
axs[0].plot(df_base['iteration'], df_base['length_mean'], 
        label='Mean Length', color='skyblue', alpha=0.5)
axs[0].plot(df_base['iteration'][9:], df_base['length_mean_ma'][9:], 
        label='Moving Average (10 iterations)', color='blue', linewidth=2)
axs[0].set_ylabel('Mean Length')
axs[0].set_xlabel('Iteration')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Plot 2: Training Loss
axs[1].set_title('v17.1 Training Loss Over Time')
axs[1].plot(df_base['iteration'], df_base['loss'], 
        label='Loss', color='lightcoral', alpha=0.5)
axs[1].plot(df_base['iteration'][9:], df_base['loss_ma'][9:], 
        label='Moving Average (10 iterations)', color='red', linewidth=2)
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Iteration')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Total iterations trained: {df_base['iteration'].max()}")
print(f"Best mean length achieved: {df_base['length_mean'].max():.2f}")
print(f"Final mean length: {df_base['length_mean'].iloc[-1]:.2f}")

# Uncomment below when you have multiple versions to compare:
# Example: Create v17.2, v17.3, v17.4 configs and train them
# Then uncomment and modify the comparison code below
