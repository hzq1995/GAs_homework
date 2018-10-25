# filename:GA_utils.py
# coding=utf-8
# created by Ziqiang Hua
import numpy as np
import matplotlib.pylab as plt
import warnings
warnings.filterwarnings('ignore')


def fun_1(x, y):
    part_up = np.sin(x) * np.sin(y)
    part_down = x * y
    return part_up/part_down
    # [-10.0000, 10.0000]
    # max


def fun_2(x1, x2):
    return x1 * np.sin(10*np.pi*x1) + x2 * np.sin(10*np.pi*x2)
    # [-1.0000, 2.0000]
    # max


def gene_coding(low, high, arr, precision=0.0001):
    gene = arr - low
    gene_bit = len(np.binary_repr(np.uint64((high-low)*(1/precision))))
    gene_str = ""
    for i in range(gene.shape[0]):
        gene_str += np.binary_repr(np.uint64(np.round(gene[i]*(1/precision))), gene_bit)
    return gene_str, gene_bit


def gene_decoding(low, gene_bit, gene_str, precision=0.0001):
    t = len(gene_str)//gene_bit
    g = np.zeros(t)
    for i in range(t):
        t2 = gene_str[i*gene_bit:(i+1)*gene_bit]
        g[i] = np.int(t2, 2)
    g = g * precision + low
    return g


def selection(g, s):
    j = 0
    N = np.uint64(np.sum(s))
    g_new = np.zeros((N, g.shape[1]))
    for i in range(s.shape[0]):
        if s[i]:
            tmp = s[i]
            while tmp:
                g_new[j] = g[i]
                j += 1
                tmp -= 1
    return g_new, N


def gene_switch(g, low, high, variation_rate=0.001, precision=0.0001):
    # crossover
    switch_table = np.arange(g.shape[0])
    np.random.shuffle(switch_table)
    g_new = np.zeros(g.shape)
    for i in range(len(switch_table)//2):
        gene_str1, gene_bit = gene_coding(low, high, g[switch_table[2*i]], precision)
        gene_str2, _ = gene_coding(low, high, g[switch_table[2*i+1]], precision)
        for j in range(len(gene_str1)):
            if np.random.uniform(0, 1) < 0.5:
                gene_str1, gene_str2 = switch_str(gene_str1, gene_str2, j)
        g_new[switch_table[2*i]] = gene_decoding(low, gene_bit, gene_str1, precision)
        g_new[switch_table[2*i+1]] = gene_decoding(low, gene_bit, gene_str2, precision)
    if len(switch_table) % 2 == 1:
        g_new[switch_table[-1]] = g[switch_table[-1]]

    # variation
    fit = fun_1(g_new[:, 0], g_new[:, 0])
    variation_num = np.uint64(np.round(gene_bit * g.shape[1] * g.shape[0] * variation_rate))
    for i in range(variation_num):
        variant_gene = np.random.randint(0, g.shape[0])
        if fit.max() != fun_1(g_new[variant_gene][0], g_new[variant_gene][1]):
            gen_str, _ = gene_coding(low, high, g_new[variant_gene], precision)
            variant_bit = np.random.randint(0, len(gen_str))
            if gen_str[variant_bit] == '0':
                gen_str = gen_str[0:variant_bit] + '1' + gen_str[variant_bit+1:]
            else:
                gen_str = gen_str[0:variant_bit] + '0' + gen_str[variant_bit+1:]
            g_new[variant_gene] = gene_decoding(low, gene_bit, gen_str, precision)

    return g_new


def switch_str(str1, str2, j):
    t_1 = str1[0:j] + str2[j] + str1[j+1:]
    t_2 = str2[0:j] + str1[j] + str2[j+1:]
    return t_1, t_2


def show_graph(arr, set_max=100, step=10, x_label='x', y_label='y'):
    x = np.arange(0, set_max, step)
    plt.plot(arr)
    plt.yticks(x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
