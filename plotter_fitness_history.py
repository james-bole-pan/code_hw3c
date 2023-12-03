import pickle
import matplotlib.pyplot as plt

with open('fitness_history.pkl', 'rb') as file:
    fitness_history = pickle.load(file)

plt.plot(fitness_history, label='Evolutionary Algorithm')
plt.xlabel('Generation')
plt.ylabel('Fitness (speed of the robot in m/s)')
plt.title('Fitness over generations')
plt.legend()
plt.show()