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
pop = 5
##################################################

np.seterr(divide="ignore")


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


def initial_position(w):
    v = []
    for i in range(w):
        v.append(Grid())
    return v


def encircling_prey(p_pre, a, Gbest):
    pp = copy.deepcopy(p_pre)
    for j in range(N_r):
        for k in range(N_r):
            r = random.uniform(0, 1)
            A_a = (2 * a * r) - a
            C = 2 * r
            D = abs((C * Gbest[j][k]) - pp[j][k])
            pp[j][k] = Gbest[j][k] - (A_a * D)
    return pp


def search_prey(p_pre, a, W_rand):
    pp = copy.deepcopy(p_pre)
    for j in range(N_r):
        for k in range(N_r):
            r = random.uniform(0, 1)
            A_a = (2 * a * r) - a
            C = 2 * r
            D = abs((C * W_rand[j][k]) - pp[j][k])
            pp[j][k] = W_rand[j][k] - (A_a * D)
    return pp


def spiral_update(p_pre, b, Gbest):
    pp = copy.deepcopy(p_pre)
    for j in range(N_r):
        for k in range(N_r):
            l = random.uniform(0, 1)
            D = abs((Gbest[j][k]) - pp[j][k])
            cos_p = 2 * math.pi * l
            pp[j][k] = (((D ** l) * (math.exp(b * l)) * math.cos(cos_p)) - Gbest[j][k])
    return pp


def update_s(vv):
    S = copy.deepcopy(vv)
    for j in range(N_r):
        for k in range(N_r):
            try:
                S[j][k] = round(1 / (1 + (math.exp(-(vv[j][k])))), 10)
            except OverflowError:
                S[j][k] = 0
    return S


def update_grid(S, so):
    sol = copy.copy(so)
    no_ones = 0
    while no_ones < N_turbines:
        for j in range(N_r):
            if no_ones == N_turbines:
                break
            for k in range(N_r):
                if no_ones == N_turbines:
                    break
                if random.uniform(0, 1) < S[j][k]:
                    sol[j][k] = 1
                    no_ones += 1
                else:
                    sol[j][k] = 0
    return sol


def Selection(G):
    fitness_arr = []
    sorted_chr = []
    for i in range(len(G)):
        fitness_arr.append(Obj_func(G[i]))
    sort = sorted(fitness_arr)

    for i in range(len(G)):
        for j in range(len(G)):
            if sort[i] == Obj_func(G[j]):
                sorted_chr.append(G[j])
                break

    return sorted_chr


def WOA(m, imax):
    global BestFitIter
    Gbest = []
    bestsol = []
    vv = initial_position(m)
    BestFitIter = []
    time_recorded = []
    best_so_far_sol = []
    best_so_far_fitness = 1000

    G = []
    for i in range(m):
        grid_layout = Grid()
        initial_solution = random_sol(grid_layout, N_turbines)
        G.append(initial_solution)
    sol = copy.deepcopy(G)
    f_Gbest = Obj_func((Selection(sol))[0])
    Gbest = (Selection(sol))[0]  # get index of fpbest
    for i in range(imax):
        for j in range(m):
            a = (2 - ((2 / imax) * i))
            Ax = abs(((2 * a * random.uniform(0, 1)) - a))
            p_rand = random.uniform(0, 1)
            if p_rand < 0.5:
                if Ax <= 1:
                    vv[j] = encircling_prey(vv[j], a, Gbest)
                    S = update_s(vv[j])
                    sol[j] = update_grid(S, sol[j])
                else:
                    G_rand = sol[random.randint(0, m - 1)]
                    vv[j] = search_prey(vv[j], a, G_rand)
                    S = update_s(vv[j])
                    sol[j] = update_grid(S, sol[j])
            else:
                vv[j] = spiral_update(vv[j], a, Gbest)
                S = update_s(vv[j])
                sol[j] = update_grid(S, sol[j])
        f_Gbest = Obj_func((Selection(sol))[0])
        Gbest = (Selection(sol))[0]  # get index of fpbest
        if f_Gbest < best_so_far_fitness:
            best_so_far_fitness = f_Gbest
            best_so_far_sol = copy.deepcopy(Gbest)
        BestFitIter.append(best_so_far_fitness)
        print(
        ("Iteration: {}, Numberof WTs: {},best Fitness:{}".format(i, N_turbines, best_so_far_fitness)))

    bestsol = copy.deepcopy(best_so_far_sol)
    plt.plot(BestFitIter)
    return bestsol

def main(unused_command_line_args):
    start = time.time()
    best = WOA(pop, Iterations)
    countt = 0
    for i in range(N_r):
       for j in range(N_r):
           if best[i][j] == 1:
               countt = countt + 1
    print('best_so_far_N_turbines ', countt)
    best_fit = Obj_func(best)
    print("best_fit", best_fit)
    fig = plt.figure()
    sca = fig.add_subplot(1, 1, 1)
    for i in range(len(best)):
       for j in range(len(best[i])):
           if best[i][j] == 1:
               sca.scatter(i, j, marker='1', s=500, label="Grid Layout")

    p_m = Wake_effect(best)
    Total_power = Power(p_m)
    print(" The Total Power is : ", Total_power)
    Total_cost = Cost(countt)
    print("The Total Cost is : ", Total_cost)
    end = time.time()
    print("Execution Time : ", end - start)
    plt.show()
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))