import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt
import time
import sys

from numpy.random import rand

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
pop = 20
P_best_c = 1.5
G_best_c = 1.5
Inertia_c = 0.798
VarSize = [1 , N_r]
VelMax = 0.5*(VarSize[1]-VarSize[0]);
VelMin = -VelMax;
##################################################

class Particle:
    def __init__(self, sz, Nt, VarSize):
        self.Position = np.zeros(sz)
        self.Velocity = np.zeros(VarSize)
        self.Cost = 0
        self.Best = BestParticle(sz, Nt, VarSize)

class BestParticle:
    def __init__(self, sz, Nt, VarSize):
        self.Position = np.zeros(sz)
        self.Cost = -np.inf


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
    P_matrix = Wake_effect(grid_layout)
    pow = Power(P_matrix)
    co = Cost(N_turbines)
    obj = co / pow
    return obj


def random_sol(N_r, N_turbines):
    grid = np.zeros((N_r, N_r))
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


def update_particles(particle,GlobalBest, nPop, sz, Nt, W, c1, c2, VelMin, VelMax):
    for i in range(nPop):
        # Update Velocity
        # particle[i].Velocity = W * particle[i].Velocity \
        #                        + c1 * np.random.rand(*VarSize) * (particle[i].Best.Position - particle[i].Position) \
        #                        + c2 * np.random.rand(*VarSize) * (GlobalBest.Position - particle[i].Position)
        particle[i].Velocity = W * particle[i].Velocity + \
                               c1 * np.multiply(rand(1,sz), particle[i].Best.Position - particle[i].Position) + \
                               c2 * np.multiply(rand(1,sz), GlobalBest.Position - particle[i].Position)

        # Apply Velocity Limits
        particle[i].Velocity = np.maximum(particle[i].Velocity, VelMin)
        particle[i].Velocity = np.minimum(particle[i].Velocity, VelMax)
        # Update Position
        r, c = np.where(particle[i].Position == 1)
        va =  np.round(particle[i].Velocity[r, c])

        row = r + va
        col = c + va
        for j in range(Nt):
            if row[j] < 0:
                row[j] = row[j] + sz
            if col[j] < 0:
                col[j] = col[j] + sz
            if row[j] >= sz:
                row[j] = row[j] - sz
            if col[j] >= sz:
                col[j] = col[j] - sz
        ind = np.ravel_multi_index((row.astype(int), col.astype(int)), (sz, sz))
        _, uniqueIdx = np.unique(ind, return_index=True)

        while len(uniqueIdx) != Nt:
            duplicateIdx = np.setdiff1d(np.arange(np.prod(ind.shape)), uniqueIdx)
            ind[duplicateIdx] = ind[duplicateIdx] + np.random.randint(-1, 2, size=duplicateIdx.shape)
            ind = np.minimum(ind, sz*sz-1)
            ind = np.maximum(ind, 0)
            _, uniqueIdx = np.unique(ind, return_index=True)

        particle[i].Position = np.zeros((sz, sz))
        particle[i].Position.flat[ind.astype(int)] = 1
        # Evaluate Cost
        particle[i].Cost = Obj_func(particle[i].Position)
        # Update Personal Best
        if particle[i].Cost < particle[i].Best.Cost:
            particle[i].Best.Position = copy.copy(particle[i].Position)
            particle[i].Best.Cost = particle[i].Cost

            # Update Global Best
            if particle[i].Best.Cost < GlobalBest.Cost:
                GlobalBest = copy.copy(particle[i].Best)

    return particle, GlobalBest

def PSO(nPop, c1, c2, imax, w):
    BestFitIter = []
    GlobalBest = BestParticle(N_r, N_turbines, VarSize)
    cost = []
    best_so_far_fitness = 100
    particle = []
    for i in range(nPop):
        p = Particle(N_r, N_turbines, VarSize)  # Create a new Particle object for each iteration

        # Initialize Position
        p.Position = random_sol(N_r, N_turbines)

        # Initialize Velocity
        p.Velocity = np.zeros(VarSize)

        # Evaluate Cost
        p.Cost = Obj_func(p.Position)
        cost.append(p.Cost)

        # Update Personal Best
        p.Best.Position = copy.copy(p.Position)
        p.Best.Cost = p.Cost

        particle.append(p)  # Add the Particle object to the particle array

    min_index = np.argmin(cost)
    GlobalBest = particle[min_index].Best
    for it in range(Iterations):
        particle, GlobalBest = update_particles(particle, GlobalBest, nPop, N_r, N_turbines, w, c1, c2, VelMin, VelMax)

        if GlobalBest.Cost < best_so_far_fitness:
            best_so_far_fitness = GlobalBest.Cost
            best_so_far_sol = copy.deepcopy(GlobalBest.Position)
            print("The Best So far solution", best_so_far_fitness)

        print( "Iteration: {}, Numberof WTs: {},best Fitness:{}".format(it + 1, N_turbines, GlobalBest.Cost))
        BestFitIter.append(best_so_far_fitness)
        plt.plot(BestFitIter)
    bestsol = copy.deepcopy(best_so_far_sol)
    return bestsol


def main(unused_command_line_args):
    start = time.time()
    best = PSO(pop, P_best_c, G_best_c, Iterations, Inertia_c)
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
    print("Execution Time : ", end - start, " sec")
    plt.show()
    return 0


if __name__ == '__main__':
    main(sys.argv)