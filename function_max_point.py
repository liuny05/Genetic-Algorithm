#-*- coding:utf-8 -*-
"""
Visualize Genetic Algorithm to find a maximum point in a function.
"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 17         # DNA length
POP_SIZE = 100        # population size
KEEP_NUM = 10         # keep KEEP_NUM individual with max fitness while selecting to avoid coverge in local optimal solution
CROSS_RATE = 0.8      # mating probability (DNA crossover)
MUTATION_RATE = 0.01  # mutation probability
N_GENERATIONS = 100   # num of generations
X_BOUND = [0, 9]      # x upper and lower bounds

# define your own function here
def f(x):
	return x + 10*np.sin(5*x) + 7*np.cos(4*x)

def decode(population):
	return X_BOUND[0] + (X_BOUND[1] - X_BOUND[0]) * (population.dot(2 ** np.arange(DNA_SIZE)[::-1]) / (2**DNA_SIZE - 1))

def random_init():
	return np.random.randint(0, 2, (POP_SIZE, DNA_SIZE))

def cal_fitness(f_value):
	return f_value - f_value.min() + 1e-3

def select(population, fitness):
	keep_idx = fitness.argsort()[-KEEP_NUM:][::-1]
	choice_idx = np.random.choice(np.arange(POP_SIZE), POP_SIZE-KEEP_NUM, replace=True, p=fitness/fitness.sum())
	return np.concatenate((population[keep_idx], population[choice_idx]))

def cross(parent, population):
	if np.random.rand() < CROSS_RATE:
		i = np.random.randint(0, POP_SIZE)
		cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)
		parent[cross_points] = population[i, cross_points]
	return parent

def mutate(child):
	for i in range(DNA_SIZE):
		if np.random.rand() < MUTATION_RATE:
			child[i] = 0 if child[i] == 1 else 0
	return child

plt.ion()
xs = np.arange(0,9,0.01)
plt.plot(xs, f(xs))

population = random_init()
step = 0
max_fs = []

while step < N_GENERATIONS:
	# calculate fitness
	f_value = f(decode(population))
	fitness = cal_fitness(f_value)

	# print staff
	max_f = f_value.max()
	max_fs.append(max_f)
	val = decode(population)[fitness.argmax()]
	print('{}: val {}, max {}'.format(step, val, max_f))
	step += 1

	# plot staff
	if 'sca' in globals(): sca.remove()
	sca = plt.scatter(decode(population), f_value, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.01)

	# select, cross and mutate
	population = select(population, fitness)
	population_copy = population.copy()
	for parent in population:
		child = cross(parent, population_copy)
		child = mutate(child)
		parent[:] = child

plt.figure()
plt.plot(range(N_GENERATIONS), max_fs)
plt.ioff(); plt.show()
