# this is the genetic algorithm I'm planning on using.

# to learn more about genetic algorithm, I'll just write it from scratch.

import math
import random
import sys

import numpy as np

from KNN import KNN

popolation_size = 20
generation_size = 4000
# fitness_func = None  # this function should return a fitness value
mutation_rate = 0.1  # how many individuals gets mutated?


def selection(population, debug=False):
    result = []
    fit_list = []
    max_fit = 0.0
    min_fit = float('inf')
    average_fit = 0.0
    best_individual = population[0]
    count = 0
    for individual in population:
        fit = fitness_func(individual)
        fit_list.append(fit)
        if fit > max_fit:
            max_fit = fit
            best_individual = individual
        if fit < min_fit:
            min_fit = fit

        average_fit += fit
        count += 1
    if debug:
        print("current best individual:", best_individual)
        print("current best fitness(1 - loss):", max_fit)
    # average_fit = average_fit / float(count)
    count = 0
    selected = 0
    while selected < len(population) / 2:
        fit = fit_list[count]
        chance = (fit - min_fit + sys.float_info.min) / (max_fit - float(min_fit) + sys.float_info.min)
        # chance = 0.1 if fit <= average_fit else 0.9
        toss = random.random()
        if toss < chance:
            result.append(population[count])
            selected += 1
        count += 1
        count %= len(population)

    return result


def cross_over(population, target_number):
    # lets mate to hit a children size of target_number
    result = []
    parents = selection(population)
    random.shuffle(parents)
    pair_num = int(math.floor(len(parents) / 2))
    male = parents[0: pair_num]
    female = parents[pair_num: pair_num * 2]
    length = len(male[0])
    count = 0
    for i in range(0, target_number, 2):
        pivot = random.randrange(1, length)
        child_1 = male[count][0:pivot] + female[count][pivot:length]
        child_2 = female[count][0:pivot] + male[count][pivot:length]
        result.append(child_1)
        result.append(child_2)
        count += 1
        count %= pair_num
    return result


def mutation(population, gene_pool):
    # mutate the entire population randomly, choosing randomly from the gene_pool for mutation
    length = len(population[0])  # total # of genes of a single individual

    for i in range(len(population)):
        if random.random() < mutation_rate:
            index = random.randrange(0, length)
            new_gene = random.choice(gene_pool)
            population[i][index] = new_gene

    return population


def first_generation(length, gene_pool, gen_size):
    result = []
    for i in range(gen_size):
        new_ind = random.choices(gene_pool, k=length)
        result.append(new_ind)
    return result


def fitness_func(individual):
    k = individual[14] + individual[15] + individual[16] + 1
    weights = np.array(individual[0:14])

    def distance_function(x, y):
        return np.sum(((x - y) * weights) ** 2)

    model = knn.get_model(k=k, dist=distance_function)
    loss = knn.get_loss(model=model)
    fit = 1 - loss
    return fit


def best_fit(population):
    # return the best fit of the current generation
    current_best = population[0]
    current_fit = 0
    for ind in population:
        new_fit = fitness_func(ind)
        if new_fit > current_fit:
            current_best = ind
            current_fit = new_fit

    print("current best fitness:", current_fit)
    print("current best value:", current_best)
    return current_best


knn = KNN(total=0.075, ratio=0.8)

pool = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# target = [2, 1, 4, 4, 3, 8, 4, 6, 9, 1, 1, 2, 3, 4,]
current_pop = first_generation(17, pool, popolation_size)  # the sum of the last 3 value (14, 15, 16) + 1 is the k value
# print(current_pop)
counter = 0

for i in range(generation_size):
    print("current generation:", i)
    sel = selection(current_pop, debug=True)
    # print("current selected size:", len(sel))
    cro = cross_over(current_pop, popolation_size - len(sel))
    new_pop = mutation(sel + cro, pool)
    current_pop = new_pop
    # print(best_fit(current_pop))
    knn.re_sample()
    # counter += 1
    # print("current generation size", len(current_pop))
    # if counter > generation_size:
    #     break

print("result:", best_fit(current_pop))
print("generations:", counter)
# print(len(selection([None] * 100)))

# print(first_generation(9, pool, 20))
