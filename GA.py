# this is the genetic algorithm I'm planning on using.

# to learn more about genetic algorithm, I'll just write it from scratch.

import math
import random
import sys

import numpy as np

from KNN import KNN

CATEGORICAL_DATA = [1, 3, 5, 6, 7, 8, 9, 13]
NUMERIC_DATA = [0, 2, 4, 10, 11, 12]


class GA:
    current_pop = []
    gene_pool = []
    population_size = 20
    generation_size = 4000
    chro_length = 14
    fitness_func = None
    on_iteration = None
    # fitness_func = None  # this function should return a fitness value
    mutation_rate = 0.1  # how many individuals gets mutated?

    calculated_cur_gen = False
    best_individual = None
    best_fitness = None
    current_fitness = []  # current fitness of all the individuals

    def __init__(self, fit_func, on_iteration, gene_pool, chro_length=14, population=20, generation=200, mutation=0.1):
        self.fitness_func = fit_func
        self.on_iteration = on_iteration
        self.gene_pool = gene_pool
        self.chro_length = chro_length
        self.population_size = population
        self.generation_size = generation
        self.mutation_rate = mutation

    def start_training(self):
        for i in range(self.generation_size):
            print("current generation:", i)
            sel = self.selection(self.current_pop)
            # print("current selected size:", len(sel))
            cro = self.cross_over(self.current_pop, self.population_size - len(sel))
            new_pop = self.mutation(sel + cro)
            self.current_pop = new_pop
            self.calculated_cur_gen = False
            self.on_iteration(self.best_individual)
            print("current best individual:", self.best_individual)
            print("current best fitness(1 - loss):", self.best_fitness)
            print("current gen finished")
            # print(best_fit(current_pop))
            # counter += 1
            # print("current generation size", len(current_pop))
            # if counter > generation_size:
            #     break

    def selection(self, population):
        result = []
        fit_list = []
        max_fit = 0.0
        min_fit = float('inf')
        average_fit = 0.0
        best_individual = population[0]
        count = 0
        for i in range(len(population)):
            individual = population[i]
            if not self.calculated_cur_gen:
                fit = self.fitness_func(individual)
            else:
                fit = self.current_fitness[i]
            fit_list.append(fit)
            if fit > max_fit:
                max_fit = fit
                best_individual = individual
            if fit < min_fit:
                min_fit = fit

            average_fit += fit
            count += 1
        self.current_fitness = fit_list
        self.calculated_cur_gen = True
        self.best_individual = best_individual
        self.best_fitness = max_fit
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

    def cross_over(self, population, target_number):
        # lets mate to hit a children size of target_number
        result = []
        parents = self.selection(population)
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

    def mutation(self, population):
        # mutate the entire population randomly, choosing randomly from the gene_pool for mutation
        length = len(population[0])  # total # of genes of a single individual

        for i in range(len(population)):
            if random.random() < self.mutation_rate:
                index = random.randrange(0, length)
                new_gene = random.choice(self.gene_pool)
                population[i][index] = new_gene

        return population

    def best_fit(self):
        # return the best fit of the current generation
        # current_best = self.current_pop[0]
        # current_fit = 0
        # for ind in self.current_pop:
        #     new_fit = self.fitness_func(ind)
        #     if new_fit > current_fit:
        #         current_best = ind
        #         current_fit = new_fit

        print("current best fitness:", self.best_fitness)
        print("current best value:", self.best_individual)
        return self.best_individual

    def first_generation(self):
        result = []
        for i in range(self.population_size):
            new_ind = random.choices(self.gene_pool, k=self.chro_length)
            result.append(new_ind)
        # return result
        self.current_pop = result


def iteration(individual):
    knn.re_sample()


def fit_func_1(individual):
    # individual = [1, 1, 0, 1, 5, 9, 1, 1, 5, 4, 8, 4, 2, 1, 9, 0, 2]
    k = individual[14] + individual[15] + individual[16] + 1
    weights = np.array(individual[0:14])

    def fit(x, y):
        diff = (x - y)
        diff[2] = diff[2] / 10000
        result = (diff * weights) ** 2
        return np.sqrt(np.sum(result))

    model = knn.get_model(k, dist=fit)
    loss = knn.get_loss(model=model)
    # print("diff2:", diff)
    return 1 - loss


# knn = KNN(total=0.05, ratio=0.8, testing=False)
#
# my_ga = GA(fit_func_1, iteration, gene_pool=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], chro_length=17)
# my_ga.first_generation()
# my_ga.start_training()
#

def fitness_func(individual):
    k = individual[16] * 2 + 1  # now only the last value is used to estimate T
    weight_hamming = np.array(individual)[CATEGORICAL_DATA]
    weight_euclid = np.array(individual)[NUMERIC_DATA]

    # weights = np.array(individual[0:14])
    # print("fitness function reached")

    # print("hamming weights:", weight_hamming)
    # print("euclid weights:", weight_euclid)
    def distance_function(x, y):
        # print(x[categorical_data])
        x_h = x[CATEGORICAL_DATA]
        # print("again", x_h)
        y_h = y[CATEGORICAL_DATA]
        x_e = x[NUMERIC_DATA]
        y_e = y[NUMERIC_DATA]
        euclid_result = euclid_dist(x_e, y_e, weight_euclid)
        hamming_result = hamming_dist(x_h, y_h, weight_hamming)
        # print("euclid distance:", euclid_result)
        # print("hamming distance:", hamming_result)
        return individual[14] * euclid_result + individual[15] * hamming_result

    model = knn.get_model(k=k, dist=distance_function)
    loss = knn.get_loss(model=model)
    fit = 1 - loss
    return fit


def hamming_dist(x, y, w):
    result = 0
    for i in range(len(x)):
        if not x[i] == y[i]:
            result += w[i]
    return result


def euclid_dist(x, y, w):
    # print("wtf", x[2])
    diff = (x - y)
    # diff[1] = diff[1] / 10000
    result = (diff * w) ** 2
    # print("diff2:", diff)
    return np.sqrt(np.sum(result))


def opt_dis_2(x, y):
    # print(x[categorical_data])
    individual = [3, 9, 0, 6, 9, 4, 2, 8, 5, 4, 1, 9, 2, 9, 2, 9, 7]
    k = individual[16] * 2 + 1  # now only the last value is used to estimate T
    weight_hamming = np.array(individual)[CATEGORICAL_DATA]
    weight_euclid = np.array(individual)[NUMERIC_DATA]
    x_h = x[CATEGORICAL_DATA]
    # print("again", x_h)
    y_h = y[CATEGORICAL_DATA]
    x_e = x[NUMERIC_DATA]
    y_e = y[NUMERIC_DATA]
    euclid_result = euclid_dist(x_e, y_e, weight_euclid)
    hamming_result = hamming_dist(x_h, y_h, weight_hamming)
    # print("euclid distance:", euclid_result)
    # print("hamming distance:", hamming_result)
    return individual[14] * euclid_result + individual[15] * hamming_result


def opt_dis(x, y):
    individual = [1, 1, 0, 1, 5, 9, 1, 1, 5, 4, 8, 4, 2, 1, 9, 0, 2]
    # k = individual[14] + individual[15] + individual[16] + 1
    weights = np.array(individual[0:14])
    diff = (x - y)
    diff[2] = diff[2] / 10000
    result = (diff * weights) ** 2
    # print("diff2:", diff)
    return np.sum(result)


knn = KNN(total=0.25, ratio=0.8, testing=True)
model = knn.get_model(k=11, dist=opt_dis)
print("model 1:", knn.get_loss(model=model, write_file="KNN_euclid_dist.result"))

# knn = KNN(total=0.25, ratio=0.8, testing=True)
model = knn.get_model(k=7, dist=opt_dis_2)
print("model 2: hamming + euclid", knn.get_loss(model=model, write_file="KNN_comp_dist.result"))

# pool = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# target = [2, 1, 4, 4, 3, 8, 4, 6, 9, 1, 1, 2, 3, 4,]
# current_pop = first_generation(17, pool, popolation_size)  # the sum of the last 3 value (14, 15, 16) + 1 is the k value
# print(current_pop)
# counter = 0

# print("result:", best_fit(current_pop))
# print("generations:", counter)
# print(len(selection([None] * 100)))

# print(first_generation(9, pool, 20))
