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
				 population_size,
				 num_bins,
				 bins,
				 dimensions, 
				 feature_dimensions,
				 crossover_rate,
				 sigma,
				 mutation_rate,
				 flag_crossover,
				 minimization):


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
		self.min_max_global = 0
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

	def mapeo(self, individual, features, fitness):
		# Mapeo de cada individuo en la celda correspondiente
		index = self.gentospace(individual, features)
		celda = self.solutions[index]
		if fitness > celda[1] or celda[1] == np.inf:
			self.solutions[index] = [individual, fitness]
			self.performances[index] = fitness
			self.replace += 1
		if self.comp(self.performances[index], self.min_max_global) == False:
			self.min_max_global = self.performances[index]
			
	def gentospace(self, individual, features):
		# Devuelve el índice del individuo en cada una de los atributos
		index = tuple()
		for i in range(self.feature_dimensions):
			b = np.digitize(features[i],self.bins[i],right = False)
			index = index + (b-1,)
		return index

	def performance(self, individual):
		# Cálculo de la performance. Se utilizar como benchmark la función Rastrigin de n-dimensiones.unif
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
			new_individual = np.clip(new_individual,0,1)
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
		plt.savefig('ejemplo_map_elites.png')
		plt.clf()