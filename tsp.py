#-*- coding:utf-8 -*-
"""
Visualize Genetic Algorithm to find the shortest path for travel sales problem.
"""
import numpy as np
import matplotlib.pyplot as plt

N_CITIES = 20         # num of cities
POP_SIZE = 100        # population size
KEEP_NUM = 5          # keep KEEP_NUM individual with max fitness while selecting
CROSS_RATE = 0.1      # mating probability (DNA crossover)
MUTATE_RATE = 0.05    # mutation probability
N_GENERATIONS = 200   # num of generations

np.random.seed(9812)

class GA():
	def __init__(self, DNA_size, pop_size, keep_num, cross_rate, mutate_rate, env):
		self.DNA_size = DNA_size
		self.pop_size = pop_size
		self.keep_num = keep_num
		self.cross_rate = cross_rate
		self.mutate_rate = mutate_rate
		self.env = env

		self.pop = np.vstack([np.random.permutation(self.DNA_size) for _ in range(self.pop_size)])

	def cal_fitness(self):
		raw_fitness = np.array([self.env.cal_distance(p) for p in self.pop])
		fitness = np.exp(self.DNA_size * 2 / raw_fitness)
		return raw_fitness, fitness

	def select(self, fitness):
		keep_idx = fitness.argsort()[-self.keep_num:][::-1]
		choice_idx = np.random.choice(np.arange(self.pop_size), self.pop_size-self.keep_num, replace=True, p=fitness/fitness.sum())
		return np.concatenate((self.pop[keep_idx], self.pop[choice_idx]))

	def cross(self, parent, pop):
		if np.random.rand() < self.cross_rate:
			i = np.random.randint(0, self.pop_size)
			cross_points = np.random.randint(0, 2, size=self.DNA_size).astype(np.bool)
			keep_city = parent[cross_points]
			swap_city = pop[i, np.isin(pop[i], keep_city, invert=True)]
			parent = np.concatenate((keep_city, swap_city))
		return parent

	def mutate(self, child):
		for i in range(self.DNA_size):
			if np.random.rand() < self.mutate_rate:
				j = np.random.randint(0, self.DNA_size)
				tmp = child[i]
				child[i] = child[j]
				child[j] = tmp
		return child

	def evolve(self):
		raw_fitness, fitness = self.cal_fitness()
		self.pop = self.select(fitness)
		pop_copy = self.pop.copy()
		for parent in self.pop:
			child = self.cross(parent, pop_copy)
			child = self.mutate(child)
			parent[:] = child
		return raw_fitness.min(), pop_copy[raw_fitness.argmin()]

class TSP():
	def __init__(self, n_cities):
		self.cities = np.random.rand(n_cities, 2)
		def euclidean(p1, p2):
			return np.sqrt(np.sum((p1 - p2)**2))
		self.city_distance = np.zeros((n_cities, n_cities))
		for i in range(n_cities):
			for j in range(i+1, n_cities):
				self.city_distance[i, j] = self.city_distance[j, i] = euclidean(self.cities[i], self.cities[j])
		plt.ion()
		
	def cal_distance(self, path):
		path_distance = 0.
		for i in range(len(path)-1):
			path_distance += self.city_distance[path[i], path[i+1]]
		return path_distance

	def plot(self, path, min_d):
		x = self.cities[:, 0][path]
		y = self.cities[:, 1][path]
		plt.cla()
		plt.scatter(self.cities[:, 0].T, self.cities[:, 1].T, s=100, c='k')
		plt.plot(x, y, 'r-')
		plt.text(-0.05, -0.05, "Min distance=%.2f" % min_d, fontdict={'size': 20, 'color': 'red'})
		plt.xlim((-0.1, 1.1))
		plt.ylim((-0.1, 1.1))
		plt.pause(0.01)

tsp = TSP(N_CITIES)
ga = GA(N_CITIES, POP_SIZE, KEEP_NUM, CROSS_RATE, MUTATE_RATE, tsp)
d_log = []
for step in range(N_GENERATIONS):
	print(step)
	min_d, best_path = ga.evolve()
	d_log.append(min_d)
	tsp.plot(best_path, min_d)

plt.figure()
plt.plot(range(N_GENERATIONS), d_log)
plt.xlabel('iter')
plt.ylabel('min distance')
plt.ioff()
plt.show()