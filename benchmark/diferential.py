import datetime
import random
import numpy as np
import math as math
import operator as op

class Differential_Evolution():

	def __init__(self,
			 iterations, 
			 population_size,
			 dimensions, 
			 cross_probability,
			 F):

		"""
		Parámetros:
		- iterations: Número de iteraciones con la que controlar el fin del bucle
		- population_size: Población inicial
		- dimensions: Dimensiones del problema
		- cross_probability: Ratio de cruce.
		- F: Ponderación de la diferencia de vectores.
		"""
		self.iterations = iterations
		self.population_size = population_size
		self.dimensions = dimensions
		self.cross_probability = cross_probability
		self.F = F
		self.best_fitness = 1000
		self.best_individual = 0

	def generate_random_solution(self):
		new_population = [np.random.uniform(low= -5.12, high= 5.12) for d in range(self.dimensions)]
		return new_population

	def generate_initial_population(self):
		initial_population = []
		for i in range(0,self.population_size):
			random_solution = self.generate_random_solution()
			initial_population.append(random_solution)
		return np.array(initial_population)

	def mutation(self, population):
		# Selección de individuo aleatorios
		index = [idx for idx in range(population_size)]
		random_index = np.random.choice(index, 3, replace=False)
		target_vectors = population[random_index]
		mutant = target_vectors[0] + self.F * (target_vectors[1] - target_vectors[2])
		np.clip(mutant, -5.12, 5.12)
		return mutant

	def recombination(self, noisy_vector, individuo):
		# Cruce binomial
		cossover_point = np.random.rand(dimensions) < cross_probability
		trial_vector = np.where(cossover_point, noisy_vector, individuo)
		return trial_vector

	def evaluate(self, individual):
		# Función Rastringin
		fitness = 10*self.dimensions
		for i in range(len(individual)):
			fitness += individual[i]**2 - (10*math.cos(2*math.pi*individual[i]))
		return fitness

	def run(self):

		tiempo_inicial = datetime.datetime.now()
		population = self.generate_initial_population()
		for i in range(iterations):
			for j in range(len(population)):
				noisy_vector = self.mutation(population)
				trial_vector = self.recombination(noisy_vector, population[j])
				if self.evaluate(population[j]) > self.evaluate(trial_vector):
					population[j] = trial_vector
				if self.best_fitness > self.evaluate(trial_vector):
					self.best_fitness = self.evaluate(trial_vector)
					self.best_individual = trial_vector
					print(self.best_fitness)
		tiempo_transcurrido = datetime.datetime.now() - tiempo_inicial
		print("Tiempo transcurrido:")
		print(tiempo_transcurrido,"\n")
		print("Mejor individuo")
		print(self.best_individual,self.best_fitness)

iterations = 3000
dimensions = 7
population_size = dimensions*10
cross_probability = 0.9
F = 0.8

de = Differential_Evolution(iterations, population_size, dimensions, cross_probability, F)
de.run()