import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
n_simulations = 1000000  # Total number of simulations
n_trades = 300  # Number of trades per simulation
win_ratios = np.linspace(0.4, 0.7, 100)  # Range of win probabilities to simulate

# Storage for results
results = []


# Conservative Martingail strategy function
def conservative_martingail(win_loss_sequence):
    optimal = np.ones(len(win_loss_sequence))
    adjusted_all = np.zeros_like(win_loss_sequence)
    cum_adjusted_all = 0
    ratio_thresh = np.arange(start=1.5, stop=3.0, step=(3.0 - 1.5) / len(win_loss_sequence))

    for i in range(len(win_loss_sequence)):
        if i == 0:
            adjusted_all[i] = win_loss_sequence[i]
        else:
            cum_adjusted_all += adjusted_all[i - 1]
            cum_optimal = optimal[i - 1]
            ratio = abs(cum_optimal / (cum_adjusted_all + 1e-6))  # Ensure positive ratio
            if ratio > ratio_thresh[i]:  # Apply capped ratio
                ratio = ratio_thresh[i]
            adjusted_all[i] = win_loss_sequence[i] * ratio

    return np.cumsum(adjusted_all)[-1]  # Return final profit/loss


# Run simulations for each win ratio
for win_ratio in win_ratios:
    final_profits = []

    for n in range(n_simulations):
        # Generate a win/loss sequence based on the win_ratio
        wins = np.random.choice([1, -1], size=n_trades, p=[win_ratio, 1 - win_ratio])

        # Apply the Conservative Martingale strategy and get the final outcome
        final_profit = conservative_martingail(wins)
        final_profits.append(final_profit)
        if n%10000==0:
            print(n)

    # Store the average final profit for this win ratio
    avg_final_profit = np.mean(final_profits)
    results.append((win_ratio, avg_final_profit))

# Convert results to numpy array for easier plotting
results = np.array(results)

# Plot win ratio vs. average final profit
plt.plot(results[:, 0], results[:, 1])
plt.xlabel('Win Ratio')
plt.ylabel('Average Final Profit')
plt.title('Performance of Conservative Martingail across Win Ratios')
plt.show()