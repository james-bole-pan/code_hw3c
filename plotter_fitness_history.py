import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('fitness_history.pkl', 'rb') as file:
    fitness_history = pickle.load(file)

print(fitness_history)

list1 = [0.00017487162618651917, 0.886348160501961, 1.1737153408763454, 1.1737153408763454, 1.1737153408763454, 1.1737153408763454, 1.1737153408763454, 1.1737153408763454, 1.1737153408763454, 1.1737153408763454, 1.1737153408763454, 1.2785653134530701, 1.9432438081024108, 2.0138469298703425, 2.0138469298703425, 2.0138469298703425]
list2 = [0.008575173543702365, 0.8458671073137976, 0.8458671073137976, 0.8458671073137976, 0.8458671073137976, 0.8458671073137976, 0.8458671073137976, 0.8458671073137976, 0.8458671073137976, 0.8458671073137976, 0.8458671073137976, 0.8458671073137976, 0.8458671073137976, 0.8458671073137976, 0.8458671073137976, 0.8458671073137976]
list3 = [0.008144020330181868, 0.7327173935956769, 1.1633109445915983, 1.1633109445915983, 1.1633109445915983, 1.1633109445915983, 1.1633109445915983, 1.1633109445915983, 1.1633109445915983, 1.1633109445915983, 1.1633109445915983, 1.1633109445915983, 1.1633109445915983, 1.1633109445915983, 1.1633109445915983, 1.1633109445915983]
list4 = [0.0057664234760678714, 1.1194377223880325, 1.1194377223880325, 1.521893954563314, 1.521893954563314, 1.521893954563314, 1.521893954563314, 1.521893954563314, 1.521893954563314, 1.521893954563314, 1.521893954563314, 1.521893954563314, 1.521893954563314, 1.521893954563314, 1.521893954563314, 1.521893954563314]

list = [list1, list2, list3, list4]

def calculate_mean_and_std(data_lists):
    # Convert the input data_lists to a NumPy array
    data_array = np.array(data_lists)

    # Calculate the mean and standard deviation along axis 0
    mean_values = np.mean(data_array, axis=0)
    std_values = np.std(data_array, axis=0)

    # Calculate the standard error (std / sqrt(n)) for each data point
    n = len(data_lists)
    std_error = std_values / np.sqrt(n)

    return mean_values, std_error

mean_values, std_error = calculate_mean_and_std(list)
x = np.arange(len(mean_values))

# Create a figure and a subplot
fig, ax = plt.subplots()

# Plot the mean values with error bars
ax.errorbar(x, mean_values, yerr=std_error, fmt='o', color='b', label='Mean with SE')
ax.plot(x, mean_values, color='b')

# Set labels for the x and y-axes and add a legend
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')
plt.title('Fitness over Generations')

# Show the plot
plt.show()


