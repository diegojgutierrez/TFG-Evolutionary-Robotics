import math as math
import numpy as np
import operator as op
import logging as logger
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import datetime

from matplotlib import pyplot as plt
from pandas import DataFrame

class MAPElites():

	def __init__(self,
				 iterations, 
				 population_size,
				 num_bins,
				 bins,
				 dimensions, 
				 feature_dimensions,
				 crossover_rate,
				 sigma,
				 mutation_rate,
				 flag_crossover,
				 minimization = True):


		"""
		Parámetros:
		- iterations: Número de iteraciones con la que controlar el fin del bucle
		- population_size: Población inicial
		- dimensions: Dimensiones del problema
		- feature_dimensions: Dimensiones del espacio de caracerísticas
		- crossover_rate: Ratio de cruce.
		- flag_crossover: Flag para activar el cruce
		- mutation_rate : Ratio de mutación
		- minimization : True = minimizar False = maximizar
		- num_bins: Número de celdas del mapa
		- bins: Vector que contiene los intervalos de las celdas
		"""

		self.iterations = iterations
		self.population_size = population_size
		self.dimensions = dimensions
		self.feature_dimensions = feature_dimensions
		self.crossover_rate = crossover_rate
		self.flag_crossover = flag_crossover
		self.mutation_rate = mutation_rate
		self.minimization = minimization
		self.num_bins = num_bins
		self.bins = bins
		self.best_individual = 0
		self.replace = 0
		self.seed = 30
		self.sigma = sigma
		self.min_global = 10000
		self.tiempo_inicial = datetime.datetime.now()
		
		if self.minimization:
			self.comp = op.lt
		else:
			self.comp = op.gt	

		"""
		Matriz de N-Dimensiones, donde N es el número de dimensiones del espacio de atributos,
		que contendrá el mapa de celdas. Cada celda tendrá un objeto de la forma: 
		list(np.array([x0,x1,...xn], perf)) , siendo n el número de variables del problema.
		Sea crea otra matriz de las misma dimensiones con la que facilitar la representación gráfica.
		"""

		self.dimensiones_matriz = (self.num_bins,)*(self.feature_dimensions)
		self.performances = np.full(self.dimensiones_matriz,np.inf)
		self.solutions = np.full(self.dimensiones_matriz,np.inf,dtype = object)
		self.solutions.fill([np.inf,np.inf])

	def mapeo(self, individual):
		# Mapeo de cada individuo en la celda correspondiente
		index = self.gentospace(individual)
		celda = self.solutions[index]
		if self.comp(self.performance(individual), celda[1]):
			self.solutions[index] = [individual, self.performance(individual)]
			self.performances[index] = self.performance(individual)
			self.replace += 1
		if self.performances[index] < self.min_global:
			self.min_global = self.performances[index]
			

	def gentospace(self, individual):
		# Devuelve el índice del individuo en cada una de los atributos
		index = tuple()
		for i in range(self.feature_dimensions):
			b = np.digitize(individual[i],self.bins,right = False)
			index = index + (b-1,)
		return index

	def future_space(self):
		# Función para calcular el valor del individuo en cada característica.
		# En este caso como se utiliza las propias variables de la función no se utiliza.
		pass

	def generate_initial_population(self):
		# Genera cada solución y la mapea
		for i in range(0,self.population_size):
			random_solution = self.generate_random_solution()
			self.mapeo(random_solution)

	def generate_random_solution(self):
		new_population = [np.random.uniform(low= -5.12, high= 5.12) for d in range(self.dimensions)]
		return new_population

	def performance(self, individual):
		# Función Rastringin
		fitness = 10*self.dimensions
		for i in range(len(individual)):
			fitness += individual[i]**2 - (10*math.cos(2*math.pi*individual[i]))
		return fitness

	def selection(self, numero_individuos):
		# Búsqueda de individuo(s) en las celdas ocupadas.
		# El candidato tendrá la siguiente forma [np.array[x0,x1...],....np.array[x0,x1]]
		condicion_busqueda = False
		individuo_aleatorio = tuple()
		for i in range(numero_individuos):
			while condicion_busqueda == False:
				candidate_index = np.random.randint(self.num_bins, size=self.feature_dimensions)
				candidate = self.solutions[tuple(candidate_index)][0]
				if self.solutions[tuple(candidate_index)][1] != np.inf:
					condicion_busqueda = True
			individuo_aleatorio = individuo_aleatorio + (candidate,)
		return individuo_aleatorio

	def mutation(self, individual):
		# Se utiliza la mutación gaussiana para añadir una pequeña mutación
		mu = 0
		if np.random.random() < self.mutation_rate:
			variation = np.random.normal(mu, self.sigma, self.dimensions)
			new_individual = individual + variation
			new_individual = np.clip(new_individual,-5.12,5.12)
		else:
			new_individual = individual
		return new_individual

	def crossover(self, individual):
		# El cruce de individuos se realiza a través del cruce uniforme
		individual1 = individual[0]
		individual2 = individual[1]
		for i in range(len(individual)):
			if np.random.random() < crossover_rate:
				individual1[i], individual2[i] = individual2[i], individual1[i]
		return individual1, individual2 

	def ploteo(self):

		# Ratio de sustitución
		ratio_subs = self.iterations/self.replace
		print(self.iterations, self.replace)
		print("Ratio de sustituciones:")
		print(ratio_subs,"\n")

		# Porcentaje de celdas ocupadas
		occupied = np.where(self.performances != np.inf)
		percentege = (len(occupied[0]) / self.performances.size)*100
		print("Porcentaje de celdas ocupadas:")
		print(percentege,"\n")

		# Mejor solución
		index_best_performances = np.where(self.performances == np.amin(self.performances))
		index_best_solution = tuple()
		for i in range(len(index_best_performances)):
			index_best_solution = index_best_solution + (tuple(index_best_performances[i]),)
		self.best_individual = self.solutions[index_best_solution]
		best_ind = self.best_individual[0]
		print("Mejor individuo:")
		print(best_ind,"\n")

		# Norma
		norma = np.linalg.norm(best_ind[0])
		print("Norma:")
		print(norma,"\n")

		# Guardar medidas
		self.vector_resultados = [norma, ratio_subs, percentege, best_ind[1]]

		# Redimensionamiento de los datos según feature dimensions.
		if self.feature_dimensions == 1:
			df = pd.DataFrame(self.performances)
			df.replace(np.inf, np.nan, inplace = True)
		if self.feature_dimensions == 2:
			df = pd.DataFrame(self.performances)
			df.replace(np.inf, np.nan, inplace = True)
		if self.feature_dimensions == 3:
			data = self.performances.reshape(self.dimensiones_matriz[0]*self.dimensiones_matriz[2],self.dimensiones_matriz[1])
			df = pd.DataFrame(data)
			df.replace(np.inf, np.nan, inplace = True)
		if self.feature_dimensions == 4:
			data = self.performances.transpose(0,2,1,3).reshape(self.num_bins**2,self.num_bins**2)
			data_2 = self.solutions.transpose(0, 2, 1, 3).reshape(self.num_bins ** 2, self.num_bins ** 2)
			df = pd.DataFrame(data)
			df.replace(np.inf, np.nan, inplace = True)
		if self.feature_dimensions == 5:
			data = self.performances.transpose(0,1,3,2,4).reshape(self.num_bins**3,self.num_bins**2)
			df = pd.DataFrame(data)
			df.replace(np.inf, np.nan, inplace = True)
		if self.feature_dimensions == 6:
			data = self.performances.transpose(0,2,4,1,3,5).reshape(self.num_bins**3,self.num_bins**3)
			df = pd.DataFrame(data)
			df.replace(np.inf, np.nan, inplace = True)

		sns.heatmap(df, yticklabels = False, xticklabels = False, square = False, cbar_kws={'label': 'Calidad'})
		plt.xlabel("Característica 2")
		plt.ylabel("Característica 1")
		plt.savefig('ejemplo_map_elites.eps')

	def run(self):
		
		self.tiempo_inicial = datetime.datetime.now()
		
		self.generate_initial_population()
		while True:
			if self.flag_crossover == True:
				individual = self.selection(numero_individuos = 2)
				ind = self.crossover(individual)[0]
				ind = self.mutation(ind)
			else:
				individual = self.selection(numero_individuos = 1)[0]
				ind = self.mutation(individual)
			self.mapeo(ind)
			tiempo_transcurrido = datetime.datetime.now() - self.tiempo_inicial
			if tiempo_transcurrido.total_seconds() >= 30:
				break

		print("Tiempo transcurrido:")
		print(tiempo_transcurrido,"\n")
		self.ploteo()
		return self.vector_resultados

# Future space
upperlimit = 5.12
lowerlimit = -5.12
dimensions = 4
feature_dimensions = 2
granulation = 0.1
bins = np.arange(lowerlimit, upperlimit, granulation)
num_bins = len(bins)
cross = False

# Parámetros del algoritmo evolutivo
crossover_rate = 0.2
mutation_rate = 0.8
sigma = 0.05
iterations = 1000
population_size = 10000

print("Parámetros:")
print("Dimensiones = ", dimensions, " Características = ", feature_dimensions, 
" Granulación = ", granulation, " Población inicial = ", population_size," Sigma = ", sigma, " Mutación = " ,mutation_rate)
mp = MAPElites(iterations, population_size, num_bins, bins, dimensions, feature_dimensions, crossover_rate, sigma,
mutation_rate, flag_crossover=cross, minimization=True)
resultado = mp.run()