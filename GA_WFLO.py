# GA Implementation
import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt
import time
import sys

vf = 16  # wind speed
length = 2000  # width and length
k = 0.094  # wake decay coff
p = 1.225  # air dinsity
Cef = 0.4  # eff
Ct = 0.88  # thrust coff
D = 63.6
N_r = int(math.floor(length / (5 * D)))

############Algorithm Parameters##################
N_turbines = 15
Iterations = 500
Mutation_p = 0.2
pop = 20
##################################################


def Grid():
    grid_layout = np.zeros((N_r, N_r))
    return grid_layout


def Cost(N_turbines):
    c = (N_turbines * ((2. / 3) + ((1. / 3) * (np.exp((-0.00174 * ((N_turbines) ** 2)))))))
    return c


def Wake_effect(grid_layout):
    A = (3.14) * ((D / 2) ** 2)
    P_matrix = np.zeros((N_r, N_r))
    Vdf = np.zeros((N_r, N_r))
    Dwk = np.zeros((N_r, N_r))
    for i in range(N_r):
        for j in range(N_r):
            if j == 0 and grid_layout[i][j] == 1:
                P_matrix[i][j] = 0.5 * p * A * Cef * (vf ** 3)  # * (10**-6) #megawatt
            else:
                if (grid_layout[i][j] == 1):
                    c_j = j
                    c_empty = 0
                    while c_j > 0:
                        if grid_layout[i][c_j - 1] == 0:
                            c_empty += 1
                        else:
                            break
                        c_j = c_j - 1
                    s = 5. * D + ((5. * D) * c_empty)
                    Dwk[i][j] = D + (2 * k * s)
                    Vdf[i][j] = vf * ((1 - (math.sqrt(1 - Ct))) * ((D / (Dwk[i][j])) ** 2))
                    P_matrix[i][j] = 0.5 * p * A * Cef * ((vf - Vdf[i][j]) ** 3)

                else:
                    Vdf[i][j] = 0
                    P_matrix[i][j] = 0
    return P_matrix


def Power(P_matrix):
    p_total = np.sum(P_matrix) * 0.001
    return p_total


def Obj_func(grid_layout):
    cou = 0
    for i in range(N_r):
        for j in range(N_r):
            if grid_layout[i][j] == 1:
                cou = cou + 1
    P_matrix = Wake_effect(grid_layout)
    pow = Power(P_matrix)
    co = Cost(cou)
    obj = co / pow
    return obj


def random_sol(grid, N_turbines):
    no_ones = 0
    while no_ones < N_turbines:
        for i in range(N_r):
            if no_ones == N_turbines:
                break
            for j in range(N_r):
                if no_ones == N_turbines:
                    break
                if random.uniform(0, 1) < 0.7 and grid[i][j] != 1:
                    grid[i][j] = 1
                    no_ones += 1
                else:
                    grid[i][j] = 0
    return grid

def mutate(indv, mutation_rate):
    if np.random.rand() < mutation_rate:
        a = np.where(indv == 0)[0]
        r = np.random.choice(a)
        a = np.where(indv == 1)[0]
        row = np.random.choice(a)
        indv[row] = 0  # flip its value
        indv[r] = 1  # flip its value
    return indv


def generate_new_population(population, fitness, mutation_rate):
    # Normalize fitness to probabilities
    fitness_prob = fitness / sum(fitness)

    # Initialize new population
    new_population = np.zeros((population.shape[0], population.shape[1]))

    # Elitism: Keep best individual from previous generation
    elite_idx = np.argmin(fitness)
    new_population[0, :] = population[elite_idx, :]

    # Loop through rest of new population
    idx = 1
    while idx < population.shape[0]:
        # Select two parents from the current population
        parent1 = population[np.random.choice(population.shape[0], p=fitness_prob), :]
        parent2 = parent1
        while np.array_equal(parent1, parent2):
            parent2 = population[np.random.choice(population.shape[0], p=fitness_prob), :]
        dist = np.sum(parent1 != parent2)

        # Generate random number of positions to be crossed over
        num_positions = np.random.randint(1, dist + 1)

        while True:
            # Generate random permutation of positions to be crossed over
            crossover_pos = np.random.choice(dist, num_positions, replace=False)

            # Perform crossover
            for i in range(num_positions):
                temp = parent1[crossover_pos[i]]
                parent1[crossover_pos[i]] = parent2[crossover_pos[i]]
                parent2[crossover_pos[i]] = temp

            # Check if both binary numbers have the same number of ones
            if np.sum(parent1) == np.sum(parent2):
                break

        # Mutate
        parent1 = mutate(parent1, mutation_rate)
        parent2 = mutate(parent2, mutation_rate)

        # Add offspring to new population
        new_population[idx, :] = parent1
        idx += 1
        if idx == population.shape[0]:
            break
        new_population[idx, :] = parent2
        idx += 1

    # Return new population
    return new_population


def GA(imax, N_chr, Mp):
    global recorded_best_fitness
    G = np.zeros((N_chr, N_r*N_r))
    best_so_far_sol = []
    best_so_far_fitness = 1000
    recorded_best_fitness = []
    c = []
    for i in range(N_chr):
        grid_layout = Grid()
        initial_solution = random_sol(grid_layout, N_turbines)
        c.append(Obj_func(initial_solution))
        G[i] = np.array(initial_solution).flatten()

    for f in range(imax):
        G = generate_new_population(G, c, Mp)
        c =[]
        for i in range(N_chr):
            c.append(Obj_func(G[i].reshape(6,6)))
        best_so_far_fitness = c[0]
        best_so_far_sol = G[i].reshape(6,6)
        recorded_best_fitness.append(best_so_far_fitness)
        plt.plot(recorded_best_fitness)
        print("Iteration: {}, Numberof WTs: {},best Fitness:{}".format(f, N_turbines, best_so_far_fitness) )
    return best_so_far_sol


def main(unused_command_line_args):
    start = time.time()
    best = GA(Iterations, pop, Mutation_p)
    best_fit = Obj_func(best)
    p_m = Wake_effect(best)
    Total_power = Power(p_m)
    Total_cost = Cost(N_turbines)
    print("best_fit", best_fit)
    fig = plt.figure()
    sca = fig.add_subplot(1, 1, 1)
    for i in range(len(best)):
      for j in range(len(best[i])):
        if best[i][j] == 1:
          sca.scatter(i, j, marker='1', s=500, label="Grid Layout")
    print(" The Total Power is : ", Total_power)
    print("The Total Cost is : ", Total_cost)
    end = time.time()
    print("Execution Time : ", end - start , " sec")
    plt.show()
    return 0


if __name__ == '__main__':
    main(sys.argv)