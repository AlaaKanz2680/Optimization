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
D = 63.6  # Diameter
N_r = int(math.floor(length / (5 * D)))

############Algorithm Parameters##################
N_turbines = 15
Iterations = 500
Ti = 1000
Tf =.001
alpha = 0.65
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

def find_best_location(grid_layout):
    bl = list(zip(*np.where(grid_layout == 0)))
    for i in range(len(bl)):
        if bl[i][0] == 0 and grid_layout[(bl[i][0]) + 1][bl[i][1]] == 0:
            return bl[i]
        elif bl[i][0] == N_r - 1 and grid_layout[(bl[i][0]) - 1][bl[i][1]] == 0:
            return bl[i]
        elif bl[i][0] != 0 and bl[i][0] != N_r - 1 and grid_layout[(bl[i][0]) + 1][bl[i][1]] == 0 and \
                grid_layout[(bl[i][0]) - 1][bl[i][1]] == 0:
            return bl[i]
    return bl[random.randint(0, len(bl) - 1)]


def remove_insert(grid_layout):
    temp = copy.copy(grid_layout)
    p_temp = Wake_effect(temp)
    a = list(zip(*np.where(p_temp == p_temp.min())))
    p1 = a[random.randint(0, len(a) - 1)]
    p2 = find_best_location(temp)
    temp[p1[0]][p1[1]] = 0
    temp[p2[0]][p2[1]] = 0
    return temp

def Obj_func(grid_layout):
    P_matrix = Wake_effect(grid_layout)
    pow = Power(P_matrix)
    co = Cost(N_turbines)
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


def SA(Ti, Tf, imax, nt, alpha):
    global record_best_fitness
    grid_layout = Grid()
    initial_solution = random_sol(grid_layout, N_turbines)
    record_best_fitness = []
    curr_solution = initial_solution
    best_so_far = initial_solution
    curr_fitness = Obj_func(initial_solution)
    best_so_far_fitness = curr_fitness
    curr_temp = Ti
    i = 0
    while i < imax:
        new_solution = random_sol(Grid(), N_turbines)
        for j in range(nt):
            new_solution = remove_insert(new_solution)
            new_fitness = Obj_func(new_solution)
            # print 'new_solution is', new_solution
            # print "cost is", Cost(new_N_turbines)

            if new_fitness <= curr_fitness:
                curr_solution = new_solution
                curr_fitness = new_fitness

            else:
                rand_num = np.random.rand()
                acc_form = np.exp(-(new_fitness - curr_fitness) / curr_temp)
                if rand_num <= acc_form:
                    curr_solution = new_solution
                    curr_fitness = new_fitness

            if curr_fitness < best_so_far_fitness:
                best_so_far = curr_solution
                # print ' best_so_far_grid' ,best_so_far
                best_so_far_fitness = curr_fitness
        if curr_temp > Tf:
            curr_temp = curr_temp * alpha
        else:
            curr_temp = Ti
        i += 1
        # print"Solution is ",  best_so_far_fitness #to test if the code stuck here
        # print "best_so_far_N_turbines ", best_so_far_N_turbines

        best_cost = Cost(N_turbines)

        print(
            ("Iteration: {}, Numberof WTs: {},best Fitness:{}".format(i, N_turbines, best_so_far_fitness)))
        record_best_fitness.append(best_so_far_fitness)
        # Ln.set_ydata(record_best_fitness)
        # Ln.set_xdata(range(len(record_best_fitness)))
        # plt.pause(0.1)

    plt.plot(record_best_fitness)
    return best_so_far


def main(unused_command_line_args):
    start = time.time()
    best = SA(Ti, Tf, Iterations, 50, alpha)
    countt = 0
    for ii in range(N_r):
        for j in range(N_r):
            if best[ii][j] == 1:
                countt = countt + 1
    best_fit = Obj_func(best)
    p_m = Wake_effect(best)
    Total_power = Power(p_m)
    Total_cost = Cost(countt)
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
    sys.exit(main(sys.argv))

