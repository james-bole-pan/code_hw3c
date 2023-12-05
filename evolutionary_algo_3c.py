import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import pickle
import copy

# Constants
g = np.array([0.0, 0.0, -9.81])  # Gravity
dt = 0.0001
k = 1000.0  # Spring constant
L0 = 1.0  # Rest length of the spring
damping = 0.75  # Damping constant
mu_s = 1.0  # Static friction coefficient
mu_k = 0.8  # Kinetic friction coefficient
half_L0 = L0/2
drop_height = 2.0
omega = 2*np.pi*2 # frequency of breathing
times_of_simulation = 10000
mutation_range_k = [1000, 1200]
mutation_range_b = [0.05, 0.075]
mutation_range_c = [0, 2*np.pi*0.1]
mass_mutation_probability = 0.2
spring_mutation_probability = 0.2
cross_over_removal_rate = 0.5
population_size = 3
generations = 2
new_mass_spring_num = 5
mass_to_mutate = 3

class Mass:
    def __init__(self, p, v, m=0.1):
        self.m = m
        self.p = np.array(p)
        self.v = np.array(v)
        self.a = np.zeros(3,dtype=float)
        self.f = np.zeros(3,dtype=float)

class Spring:
    def __init__(self, L0, k, m1, m2):
        self.L0 = L0
        self.k = k
        self.m1 = m1
        self.m2 = m2

class Individual:
    def __init__(self):
        masses = [
            Mass([-half_L0, -half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),  # 0
            Mass([half_L0, -half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 1
            Mass([-half_L0, -half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 2
            Mass([half_L0, -half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 3
            Mass([-half_L0, half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 4
            Mass([half_L0, half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 5
            Mass([-half_L0, half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 6
            Mass([half_L0, half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),      # 7
            Mass([-half_L0, half_L0 + L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),  # 8
            Mass([half_L0, half_L0 + L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 9
            Mass([-half_L0, half_L0 + L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 10
            Mass([half_L0, half_L0 + L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 11
        ]
        short_diag_length = np.sqrt(2 * L0**2)
        long_diag_length = np.sqrt(3 * L0**2)

        springs = [
            Spring(L0, k, masses[0], masses[1]),  # Base square
            Spring(L0, k, masses[1], masses[3]),
            Spring(L0, k, masses[3], masses[2]),
            Spring(L0, k, masses[2], masses[0]),
            Spring(L0, k, masses[4], masses[5]),  # Top square
            Spring(L0, k, masses[5], masses[7]),
            Spring(L0, k, masses[7], masses[6]),
            Spring(L0, k, masses[6], masses[4]),
            Spring(L0, k, masses[0], masses[4]),  # Vertical edges
            Spring(L0, k, masses[1], masses[5]),
            Spring(L0, k, masses[2], masses[6]),
            Spring(L0, k, masses[3], masses[7]),
            Spring(short_diag_length, k, masses[0], masses[3]),
            Spring(short_diag_length, k, masses[1], masses[2]),
            Spring(short_diag_length, k, masses[4], masses[7]),
            Spring(short_diag_length, k, masses[5], masses[6]),
            Spring(short_diag_length, k, masses[0], masses[5]),
            Spring(short_diag_length, k, masses[1], masses[4]),
            Spring(short_diag_length, k, masses[2], masses[7]),
            Spring(short_diag_length, k, masses[3], masses[6]),
            Spring(short_diag_length, k, masses[1], masses[7]),
            Spring(short_diag_length, k, masses[0], masses[6]),
            Spring(short_diag_length, k, masses[3], masses[5]),
            Spring(short_diag_length, k, masses[2], masses[4]),
            Spring(long_diag_length, k, masses[0], masses[7]),
            Spring(long_diag_length, k, masses[1], masses[6]),
            Spring(long_diag_length, k, masses[2], masses[5]),
            Spring(long_diag_length, k, masses[3], masses[4]),
            Spring(L0, k, masses[8], masses[9]), 
            Spring(L0, k, masses[9], masses[11]),
            Spring(L0, k, masses[11], masses[10]),
            Spring(L0, k, masses[10], masses[8]),
            Spring(L0, k, masses[6], masses[10]),
            Spring(L0, k, masses[7], masses[11]),
            Spring(L0, k, masses[4], masses[8]),
            Spring(L0, k, masses[5], masses[9]),
            Spring(short_diag_length, k, masses[6], masses[11]),
            Spring(short_diag_length, k, masses[7], masses[10]),
            Spring(short_diag_length, k, masses[4], masses[9]),
            Spring(short_diag_length, k, masses[5], masses[8]),
            Spring(short_diag_length, k, masses[4], masses[10]),
            Spring(short_diag_length, k, masses[5], masses[11]),
            Spring(short_diag_length, k, masses[6], masses[8]),
            Spring(short_diag_length, k, masses[7], masses[9]),
            Spring(short_diag_length, k, masses[9], masses[10]),
            Spring(short_diag_length, k, masses[8], masses[11]),
            Spring(long_diag_length, k, masses[6], masses[9]),
            Spring(long_diag_length, k, masses[7], masses[8]),
            Spring(long_diag_length, k, masses[4], masses[11]),
            Spring(long_diag_length, k, masses[5], masses[10])
        ]
        self.masses = masses
        self.springs = springs
        self.a_dict = {}
        for spring in springs:
            self.a_dict[spring] = spring.L0
        self.b_dict = {spring:0.0 for spring in springs}
        self.c_dict = {spring:0.0 for spring in springs}
        self.k_dict = {spring:k for spring in springs}

    def set_a_dict(self, a_dict):
        self.a_dict = a_dict

    def set_b_dict(self, b_dict):
        self.b_dict = b_dict
    
    def set_c_dict(self, c_dict):
        self.c_dict = c_dict

    def set_k_dict(self, k_dict):
        self.k_dict = k_dict

    def add_mass(self, mass):
        self.masses.append(mass)
    
    def add_spring(self, spring):
        self.springs.append(spring)
        self.a_dict[spring] = spring.L0
        self.b_dict[spring] = 0.0
        self.c_dict[spring] = 0.0
        self.k_dict[spring] = k

    def remove_spring(self, spring):
        self.springs.remove(spring)
        del self.a_dict[spring]
        del self.b_dict[spring]
        del self.c_dict[spring]
        del self.k_dict[spring]

    def remove_mass(self, mass):
        self.masses.remove(mass)
        springs_to_remove = []

        for spring in self.springs:
            if spring.m1 == mass or spring.m2 == mass:
                springs_to_remove.append(spring)

        for spring in springs_to_remove:
            self.remove_spring(spring)

    def get_fitness(self):
        individual = copy.deepcopy(self)
        masses = individual.masses
        springs = individual.springs
        initial_center_of_mass = p_center_of_mass(masses)
        t = 0
        for _ in range(times_of_simulation):
            simulation_step(masses, springs, t, individual.a_dict, individual.b_dict, individual.c_dict, individual.k_dict)
            t += dt
        final_center_of_mass = p_center_of_mass(masses)
        displacement = final_center_of_mass - initial_center_of_mass
        speed = np.linalg.norm(displacement[:2]) / (times_of_simulation * dt) # only care about horizontal distance
        return speed
    
def add_random_mass(individual):
    # Add a random mass to the individual
    mass_p = np.random.rand(3) * 1.5
    new_mass = Mass(mass_p, np.zeros(3,dtype=float))
    print("Random mass position: ", mass_p)
    individual.add_mass(new_mass)
    if len(individual.masses) >= new_mass_spring_num:
        random_masses = random.sample(individual.masses, new_mass_spring_num)
    else:
        random_masses = individual.masses
    for mass in random_masses:
        distance = np.linalg.norm(new_mass.p - mass.p)
        new_spring = Spring(distance, k, new_mass, mass)
        individual.add_spring(new_spring)

def remove_random_mass(individual):
    # Remove a random spring from the individual
    if len(individual.masses) > 0:
        mass = random.choice(individual.masses)
        individual.remove_mass(mass)

def get_floor_tile():
    floor_size = 2.5
    return [[-floor_size, -floor_size, 0], 
            [floor_size, -floor_size, 0], 
            [floor_size, floor_size, 0], 
            [-floor_size, floor_size, 0]]

def simulation_step(masses, springs, t, a_dict, b_dict, c_dict, k_dict):
    # Reset forces on each mass
    for mass in masses:
        mass.f = np.zeros(3, dtype=float)
        mass.f += mass.m * g  # Gravity

    # Calculate spring forces
    for spring in springs:
        a = a_dict[spring]
        b = b_dict[spring]
        c = c_dict[spring]
        spring.k = k_dict[spring]
        spring.L0 = a + b*np.sin(omega*t+c) 

        delta_p = spring.m1.p - spring.m2.p
        delta_length = np.linalg.norm(delta_p)
        if delta_length == 0:
            direction = np.zeros(3, dtype=float)
        else:
            direction = delta_p / delta_length
        force_magnitude = spring.k * (delta_length - spring.L0)
        force = force_magnitude * direction

        # Apply spring force to masses
        spring.m1.f -= force
        spring.m2.f += force

    # tally friction
    for mass in masses:
        if mass.p[2] > 0:
            continue
        F_n = mass.m * g[2]
        F_H = np.linalg.norm(mass.f[:2])
        direction = mass.f[:2] / F_H
        if F_n < 0:
            if F_H<=-mu_s*F_n:
                mass.f[:2] = np.zeros(2)
            else:
                mass.f[:2] += -abs(mu_k*F_n)*direction

    # Update positions and velocities for each mass
    for mass in masses:
        mass.a = mass.f / mass.m
        mass.v += mass.a * dt
        mass.p += mass.v * dt

        # Simple collision with the ground
        if mass.p[2] < 0:
            mass.p[2] = 0
            mass.v[2] = -damping * mass.v[2]  # Some damping on collision

def mutation(individual,mass_mutation_probability,spring_mutation_probability):
    for i in range(0, mass_to_mutate):
        if random.random() < mass_mutation_probability:
            add_random_mass(individual)
        if random.random() < mass_mutation_probability:
            remove_random_mass(individual)

    b_dict = individual.b_dict
    c_dict = individual.c_dict
    k_dict = individual.k_dict
    for spring in individual.springs:
        if random.random() < spring_mutation_probability:
            b_dict[spring] = np.random.uniform(mutation_range_b[0], mutation_range_b[1])
            c_dict[spring] = np.random.uniform(mutation_range_c[0], mutation_range_c[1])
            k_dict[spring] = np.random.uniform(mutation_range_k[0], mutation_range_k[1])
    individual.set_b_dict(b_dict)
    individual.set_c_dict(c_dict)
    individual.set_k_dict(k_dict)
    return individual

def crossover(individual1, individual2):
    # Combine masses and springs from both individuals
    combined_masses = individual1.masses + individual2.masses
    combined_springs = individual1.springs + individual2.springs
    combined_a_dict = {**individual1.a_dict, **individual2.a_dict}
    combined_b_dict = {**individual1.b_dict, **individual2.b_dict}
    combined_c_dict = {**individual1.c_dict, **individual2.c_dict}
    combined_k_dict = {**individual1.k_dict, **individual2.k_dict}

    unique_masses = remove_duplicate_masses(combined_masses)

    # Create a new individual with the combined and filtered attributes
    new_individual = Individual()
    new_individual.masses = unique_masses
    
    new_springs = []
    # add all springs that have both masses in the new individual
    for spring in combined_springs:
        for mass in unique_masses:
            if np.array_equal(spring.m1.p, mass.p) or np.array_equal(spring.m2.p, mass.p):
                new_springs.append(spring)
                break

    # remove springs with duplicated starting and ending mass locations
    positions = set()
    unique_springs = []
    for spring in new_springs:
        pos_tuple = tuple(tuple(mass.p) for mass in [spring.m1, spring.m2])
        if pos_tuple not in positions:
            positions.add(pos_tuple)
            unique_springs.append(spring)

    for spring in unique_springs:
        if spring.m1 not in unique_masses:
            for mass in unique_masses:
                if np.array_equal(spring.m1.p, mass.p):
                    spring.m1 = mass
                    break
        if spring.m2 not in unique_masses:
            for mass in unique_masses:
                if np.array_equal(spring.m2.p, mass.p):
                    spring.m2 = mass
                    break

    new_individual.springs = unique_springs
    new_a_dict = {}
    new_b_dict = {}
    new_c_dict = {}
    new_k_dict = {}
    for spring in unique_springs:
        new_a_dict[spring] = combined_a_dict[spring]
        new_b_dict[spring] = combined_b_dict[spring]
        new_c_dict[spring] = combined_c_dict[spring]
        new_k_dict[spring] = combined_k_dict[spring]
    new_individual.set_a_dict(new_a_dict)
    new_individual.set_b_dict(new_b_dict)
    new_individual.set_c_dict(new_c_dict)
    new_individual.set_k_dict(new_k_dict)

    # randomly remove 50% of the masses
    for mass in unique_masses:
        if random.random() < cross_over_removal_rate:
            new_individual.remove_mass(mass)
    return new_individual

def remove_duplicate_masses(masses):
    unique_masses = []
    positions = set()
    for mass in masses:
        pos_tuple = tuple(mass.p)
        if pos_tuple not in positions:
            positions.add(pos_tuple)
            unique_masses.append(mass)
    return unique_masses

def find_mass(ref_mass, mass_list):
    # Helper function to find a matching mass in the new list by position
    for mass in mass_list:
        if np.array_equal(ref_mass.p, mass.p):
            return mass
    return None

population = [Individual() for _ in range(population_size)]

for individual in population:
    b_dict = {}
    c_dict = {}
    k_dict = {}

    for spring in individual.springs:
        b_dict[spring] = np.random.uniform(mutation_range_b[0], mutation_range_b[1])
        c_dict[spring] = np.random.uniform(mutation_range_c[0], mutation_range_c[1])
        k_dict[spring] = np.random.uniform(mutation_range_k[0], mutation_range_k[1])

    individual.set_b_dict(b_dict)
    individual.set_c_dict(c_dict)
    individual.set_k_dict(k_dict)

for individual in population:
    mutation(individual, 0.5, 0.5)

def p_center_of_mass(masses):
    return sum([mass.m * mass.p for mass in masses]) / sum([mass.m for mass in masses]) +0.000001

fitness_history = []
fitnesses = [individual.get_fitness() for individual in population]
max_fitness = max(fitnesses)
fitness_history.append(max_fitness)
best_individual = population[fitnesses.index(max_fitness)]

with open("dot_plot_data.txt", "w") as f:
    f.write("")

for i in range(generations):
    # select parents
    children = []
    while len(children) < population_size:
        parent1 = random.choices(population, weights=fitnesses)[0]
        parent2 = random.choices(population, weights=fitnesses)[0]

        parent1 = mutation(parent1, mass_mutation_probability, spring_mutation_probability)
        parent2 = mutation(parent2, mass_mutation_probability, spring_mutation_probability)

        child = crossover(parent1, parent2)
        child = mutation(child, mass_mutation_probability, spring_mutation_probability)
        
        springs_to_remove = []
        for spring in child.springs:
            if spring.m1 not in child.masses or spring.m2 not in child.masses:
                springs_to_remove.append(spring)

        for spring in springs_to_remove:
            child.remove_spring(spring)

        #remove all masses not connected to any spring
        masses_to_remove = []
        for mass in child.masses:
            if mass not in [spring.m1 for spring in child.springs] and mass not in [spring.m2 for spring in child.springs]:
                masses_to_remove.append(mass)

        for mass in masses_to_remove:
            child.remove_mass(mass)
        children.append(child)
    
    population = children
    
    #checking the position of every mass
    for individual in population:
        for mass in individual.masses:
            print("Mass position", mass.p)
            
    # evaluate fitness
    fitnesses = [individual.get_fitness() for individual in population]
    current_max_fitness = max(fitnesses)
    current_best_individual = population[fitnesses.index(current_max_fitness)]
    if current_max_fitness > max_fitness:
        max_fitness = current_max_fitness
        best_individual = current_best_individual
    fitness_history.append(max_fitness)
    print("Generation ", i, " fitnesses: ", fitnesses)
    with open("dot_plot_data.txt", "a") as f:
        f.write(str(fitnesses) + "\n")

print("Best individual: ", best_individual)
print("Best fitness: ", max_fitness)

with open("fitness_history.pkl", "wb") as f:
    pickle.dump(fitness_history, f)

with open("best_individual.pkl", "wb") as f:
    pickle.dump(best_individual, f)