import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

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
mutation_range_b = [0.2, 0.3]
mutation_range_c = [0, 2*np.pi*0.1]
mass_mutation_probability = 0.5
spring_mutation_probability = 0.2
crossover_probability = 0.5
population_size = 2
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

def add_random_mass(individual):
    # Add a random mass to the individual
    new_mass = Mass(np.random.rand(3), np.random.rand(3))
    individual.add_mass(new_mass)
    random_masses = random.sample(individual.masses, new_mass_spring_num)
    for mass in random_masses:
        distance = np.linalg.norm(new_mass.p - mass.p)
        new_spring = Spring(distance, k, new_mass, mass)
        individual.add_spring(new_spring)

def remove_random_mass(individual):
    # Remove a random spring from the individual
    mass = random.choice(individual.masses)
    individual.remove_mass(mass)

def get_floor_tile():
    floor_size = 2.5
    return [[-floor_size, -floor_size, 0], 
            [floor_size, -floor_size, 0], 
            [floor_size, floor_size, 0], 
            [-floor_size, floor_size, 0]]

t = 0
def simulation_step(masses, springs, dt, a_dict, b_dict, c_dict, k_dict):
    global t
    t += dt

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
                print("static friction, ", mass.f)
            else:
                mass.f[:2] += -abs(mu_k*F_n)*direction
                print("kinetic friction, ", mass.f)

    # Update positions and velocities for each mass
    for mass in masses:
        mass.a = mass.f / mass.m
        mass.v += mass.a * dt
        mass.p += mass.v * dt

        # Simple collision with the ground
        if mass.p[2] < 0:
            mass.p[2] = 0
            mass.v[2] = -damping * mass.v[2]  # Some damping on collision

def mutation(individual):
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
    # Crossover the two individuals
    b_dict1 = individual1.b_dict
    c_dict1 = individual1.c_dict
    k_dict1 = individual1.k_dict
    b_dict2 = individual2.b_dict
    c_dict2 = individual2.c_dict
    k_dict2 = individual2.k_dict
    for spring in individual1.springs:
        if random.random() < crossover_probability:
            b_dict1[spring] = b_dict2[spring]
            c_dict1[spring] = c_dict2[spring]
            k_dict1[spring] = k_dict2[spring]
    individual1.set_b_dict(b_dict1)
    individual1.set_c_dict(c_dict1)
    individual1.set_k_dict(k_dict1)
    return individual1

I = Individual()

b_dict = {}
c_dict = {}
k_dict = {}

for spring in I.springs:
    b_dict[spring] = np.random.uniform(mutation_range_b[0], mutation_range_b[1])
    c_dict[spring] = np.random.uniform(mutation_range_c[0], mutation_range_c[1])
    k_dict[spring] = np.random.uniform(mutation_range_k[0], mutation_range_k[1])

I.set_b_dict(b_dict)
I.set_c_dict(c_dict)
I.set_k_dict(k_dict)

I = mutation(I)

masses = I.masses
springs = I.springs
a_dict = I.a_dict
b_dict = I.b_dict
c_dict = I.c_dict
k_dict = I.k_dict

# Visualization setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize 8 points for the cube's vertices
points = [ax.plot([], [], [], 'ro')[0] for _ in range(len(masses))]

# Initialize 12 lines for the springs
lines = [ax.plot([], [], [], 'b-')[0] for _ in range(len(springs))] 
shadows = [ax.plot([], [], [], 'k-')[0] for _ in range(len(springs))] 

floor_tile_collection = Poly3DCollection([get_floor_tile()], color='gray', alpha=0.5)
ax.add_collection3d(floor_tile_collection)

ax.set_xlim([-2, 2]) 
ax.set_ylim([-2, 2])
ax.set_zlim([0, 4])
ax.set_xlabel('X')
ax.set_ylabel('Y')  
ax.set_zlabel('Z')
ax.set_title('Dropping and Bouncing Cube in 3D')

def init():
    for point in points:
        point.set_data([], [])
        point.set_3d_properties([])
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    for shadow in shadows:
        shadow.set_data([], [])
        shadow.set_3d_properties([])
    return points + lines + shadows

def animate(i):
    for _ in range(300):
        simulation_step(masses, springs, dt, a_dict, b_dict, c_dict, k_dict)
    
    for mass, point in zip(masses, points):
        x, y, z = mass.p
        point.set_data([x], [y])
        point.set_3d_properties([z])  # Setting the Z value for 3D

    # Update the spring lines
    for spring, line in zip(springs, lines):
        x_data = [spring.m1.p[0], spring.m2.p[0]]
        y_data = [spring.m1.p[1], spring.m2.p[1]]
        z_data = [spring.m1.p[2], spring.m2.p[2]]
        line.set_data(x_data, y_data)
        line.set_3d_properties(z_data)
    
    # Update the shadow lines
    for spring, shadow in zip(springs, shadows):
        x_data = [spring.m1.p[0], spring.m2.p[0]]
        y_data = [spring.m1.p[1], spring.m2.p[1]]
        z_data = [0, 0]
        shadow.set_data(x_data, y_data)
        shadow.set_3d_properties(z_data)
        
    return points + lines + shadows

ani = animation.FuncAnimation(fig, animate, frames=1000, init_func=init, blit=False, interval=5)

plt.show()