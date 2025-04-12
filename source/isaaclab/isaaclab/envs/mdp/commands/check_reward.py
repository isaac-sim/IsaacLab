import torch
import matplotlib.pyplot as plt
import math

# Define standard deviation
std = 0.1

# Generate a range of joint_pos_error values
joint_pos_error_range = torch.linspace(0, 1.0, 1000)

# Calculate rewards for the range of joint_pos_error values
rewards = 1.1 * torch.exp(-joint_pos_error_range / std**2) - 0.1

# Convert to numpy for plotting
joint_pos_error_range = joint_pos_error_range.numpy()
rewards = rewards.numpy()

# Plot the graph
plt.figure(figsize=(10, 5))
plt.plot(joint_pos_error_range, rewards, label='Reward')
plt.xlabel('Joint Position Error')
plt.ylabel('Reward')
plt.title('Reward Function vs Joint Position Error')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
