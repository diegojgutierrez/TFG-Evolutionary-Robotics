import math
import numpy as np
import operator as op


def generate_random_solution(low_limit, high_limit, dimensions):
    new_population = np.random.uniform(low=low_limit, high=high_limit, size = dimensions)
    return new_population

def generate_initial_population(low_limit, high_limit, dimensions, population_size):
    initial_population = []
    for i in range(0, population_size):
        random_solution = generate_random_solution(
            low_limit, high_limit, dimensions)
        initial_population.append(random_solution)
    return np.array(initial_population)

def mutation(low_limit, high_limit, population, population_size, F):
    # Selección de individuo aleatorios
    index = [idx for idx in range(population_size)]
    random_index = np.random.choice(index, 3, replace=False)
    target_vectors = population[random_index]
    mutant = target_vectors[0] + F * \
        (target_vectors[1] - target_vectors[2])
    mutant = np.clip(mutant, low_limit, high_limit)
    return mutant

def recombination(noisy_vector, individuo, dimensions, cross_probability):
    cossover_point = np.random.rand(dimensions) < cross_probability
    trial_vector = np.where(cossover_point, noisy_vector, individuo)
    return trial_vector
