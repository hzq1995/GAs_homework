# filename:GA_main.py
# coding=utf-8
# created by Ziqiang Hua
from GA_utils import *


def subject_1():
    N = 60  # Original Population
    total_iteration = 30
    G_f = np.random.uniform(-10, 10, (N, 2))
    N_record = np.zeros(total_iteration)
    fitness_record = np.zeros(total_iteration)
    fitness_max = 0
    G_record = 0
    for i in range(total_iteration):
        # print('Iteration'+str(i+1))
        fitness_value = fun_1(G_f[:, 0], G_f[:, 1])
        if fitness_max < fitness_value.max():
            fitness_max = fitness_value.max()
        # fitness_record[i] = np.mean(fitness_value)
        fitness_record[i] = fitness_max
        fitness_value = fitness_value - fitness_value.min() + 0.0001

        N_record[i] = N
        f_sum = np.sum(fitness_value)
        son_num = np.round(N * fitness_value/f_sum)  # reproduce

        G_new, N = selection(G_f, son_num)   # selection 
        G_f = gene_switch(G_new, -10, 10, variation_rate=0, precision=20/(2**18-1))  # crossover and variation
        G_record = np.mean(G_f, axis=0)

        if 1 - fitness_record[i] <= 1e-3:
            break
    print(i, fitness_max, fitness_record[i])
    # show_graph(fitness_record[0:i], 1.1, 0.2, 'Generation', 'Best Individual Fitness')
    # show_graph(N_record[0:i], 150, 10, 'Generation', 'Population Number')  
    return i


def subject_2():
    N = 500  # Original Population
    total_iteration = 150
    G_f = np.random.uniform(-1, 2, (N, 2))
    N_record = np.zeros(total_iteration)
    std_record = np.zeros(total_iteration)
    fitness_record = np.zeros(total_iteration)
    fitness_max = 0
    for i in range(total_iteration):
        # print('Iteration'+str(i+1))

        fitness_value = fun_2(G_f[:, 0], G_f[:, 1])
        if fitness_max < fitness_value.max():
            fitness_max = fitness_value.max()
        # fitness_record[i] = np.mean(fitness_value)
        fitness_record[i] = fitness_max
        fitness_value -= fitness_value.min() - 0.0001
        std_record[i] = np.std(fitness_value)

        N_record[i] = N
        f_sum = np.sum(fitness_value)
        son_num = np.round(N * fitness_value/f_sum)  # reproduce

        G_new, N = selection(G_f, son_num)   # selection
        G_f = gene_switch(G_new, -1, 2, variation_rate=0.0005, precision=3/(2**16-1))  # crossover and variation
        G_record = np.mean(G_f, axis=0)

        if abs(3.7 - fitness_record[i]) <= 1e-3:
            break

    print(i, fitness_max, fitness_record[i])
    # show_graph(fitness_record[:i], 4.1, 0.5, 'Generation', 'Best Individual Fitness')
    # show_graph(N_record[:i], 700, 50, 'Generation', 'Population Num')
    return i


def main():
    M = 50
    s1 = np.arange(M)
    s2 = np.arange(M)
    print('subject1 start!')
    for i in range(M):
        s1[i] = subject_1()
    print('subject2 start!')
    for i in range(M):
        s2[i] = subject_2()
    print('Average Generation:')
    print(np.mean(s1))
    print(np.mean(s2))

main()

