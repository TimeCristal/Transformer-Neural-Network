import numpy as np
import matplotlib.pyplot as plt
# random from 0 to 1
all_wins = np.random.random(300)
all_max_win = np.ones_like(all_wins)/2
win = np.random.random(200)
lose = -np.random.random(100)
all = np.append(win, lose)
arr_idx = np.arange(len(all))
np.random.shuffle(arr_idx)
all = all[arr_idx]
plt.plot(all)
plt.show()

plt.plot(np.cumsum(all))
plt.show()
plt.plot(np.cumsum(all_wins))
plt.show()
optimal = np.cumsum(all_max_win)
plt.plot(optimal)
plt.show()

coef_list = []
for idx in range(1, 299):
    coef = optimal[idx]-all[idx]
    coef_list.append(coef)
coefficients = np.array(coef_list)

plt.plot(coefficients)
plt.show()
#%%
# Step 1: Calculate the cumulative sum of all
cum_all = np.cumsum(all)

# Step 2: Calculate the ratio of optimal to cum_all (element-wise division)
# Adding a small constant to avoid division by zero
ratio = optimal[:len(cum_all)] / (cum_all + 1e-6)

# Step 3: Multiply each element in 'all' by the ratio to adjust it to follow 'optimal'
adjusted_all = all * ratio

# Plotting the adjusted "all"
plt.plot(adjusted_all)
plt.title("Adjusted all")
plt.show()

# Cumulative sum of adjusted "all" to see how it follows the optimal curve
cum_adjusted_all = np.cumsum(adjusted_all)
plt.plot(cum_adjusted_all, label="Adjusted cumulative")
plt.plot(optimal, label="Optimal", linestyle='--')
plt.legend()
plt.title("Adjusted cumulative sum of all vs. Optimal")
plt.show()
#%%
plt.plot(ratio, label="ratio")
plt.legend()
plt.title("ratio")
plt.show()
#%%
# Step-by-step adjustment to make "all" follow "optimal"
ratio_tresh =  np.arange(start=1.5, stop=3.0, step=1.5/len(all))
adjusted_all = np.zeros_like(all)  # Initialize adjusted "all" array
cum_adjusted_all = 0  # Variable to hold cumulative sum of adjusted values
ratio_list = []
# Iterate through each element in "all" and adjust it
for i in range(len(all)):
    if i == 0:
        # No adjustment needed for the first step, initialize with same value
        adjusted_all[i] = all[i]
    else:
        # Cumulative sums up to the current step
        cum_adjusted_all += adjusted_all[i - 1]
        cum_optimal = optimal[i - 1]

        # Calculate the ratio between optimal cumulative sum and adjusted cumulative sum
        ratio = abs(cum_optimal / (cum_adjusted_all + 1e-6))  # Add small constant to avoid division by zero
        if ratio > ratio_tresh[i]:
            ratio=ratio_tresh[i]
        ratio_list.append(ratio)
        # Adjust the current value of "all" using the ratio
        adjusted_all[i] = all[i] * ratio

# Plotting the step-adjusted "all"
plt.plot(adjusted_all, label="Adjusted all")
plt.title("Adjusted 'all' (step-by-step adjustment)")
plt.show()

# Plot cumulative sum of adjusted "all" vs. optimal
cum_adjusted_all_final = np.cumsum(adjusted_all)
plt.plot(cum_adjusted_all_final, label="Adjusted cumulative")
plt.plot(optimal, label="Optimal", linestyle='--')
plt.legend()
plt.title("Adjusted cumulative sum of all vs. Optimal (step-by-step)")
plt.show()

plt.plot(ratio_list, label="ratio")
plt.legend()
plt.title("ratio")
plt.show()