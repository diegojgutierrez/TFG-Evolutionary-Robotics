import diff
import map_elites
import numpy as np
from map_elites import MAPElites

class MAPElitesSimulation:
    def __init__(self, config, robot, environment):
        self.index = 0
        self.visual_index = 0
        self.steps = 0
        self.iteration = 0
        self.map = False
        self.finished = False
        self.visualize = False
        self.robot = robot
        self.config = config
        self.environment = environment
        self.dimensions = self.robot.nn.size

        self.max_fitness = []
        self.mean_fitness = []
        self.deviation_fitness = []

        self.robot.weights = diff.generate_initial_population(self.config.low, self.config.high, self.dimensions, self.config.population_size)
        self.current_weights = self.robot.weights[0]

        bins = [np.arange(self.config.lowerlimit, self.config.upperlimit, self.config.granulation)]*2
        num_bins = len(bins[0])
        self.mp = MAPElites(self.config.population_size, num_bins, bins, self.dimensions, self.config.feature_dimensions, self.config.crossover_rate, self.config.sigma,
        self.config.mutation_rate, flag_crossover=self.config.cross, minimization=True)

    def evolution(self):
        if self.mp.flag_crossover == True:
            individual = self.mp.selection(numero_individuos = 2)
            ind = self.mp.crossover(individual)[0]
            self.robot.current_weights = self.mp.mutation(ind)
        else:
            individual = self.mp.selection(numero_individuos = 1)[0]
            self.robot.current_weights = self.mp.mutation(individual)

        self.robot.nn.update_weights(self.robot.current_weights)

    def evaluation(self):
        self.mp.mapeo(self.robot.current_weights, self.feature_space(), self.calculate_fitness())
        self.robot.fitness = self.mp.performances

        if self.map:
            # En el caso de MAP-Elites es necesario quitar los np.inf que cubren la matriz
            self.mean_fitness.append(np.mean(self.robot.fitness[self.robot.fitness != np.inf]))
            self.deviation_fitness.append(np.std(self.robot.fitness[self.robot.fitness != np.inf]))
            self.max_fitness.append(np.amax(self.robot.fitness[self.robot.fitness != np.inf]))

    def calculate_fitness(self):

        v_mean = (self.robot.sum_vel/self.steps)
        v_diff = 1 - np.sqrt(np.absolute(self.robot.sum_v_l/(self.steps + 1) - self.robot.sum_v_r/(self.steps + 1)))
        fitness = v_mean * v_diff * self.steps/self.config.evaluation_steps

        return  fitness
    
    def feature_space(self):
        features = [self.robot.x_final, self.robot.y_final]
        return features

    def select_best(self):
        # Selección de mejores individuos para su representación
        self.best_index = np.argwhere((self.robot.fitness > 0.8*self.max_fitness[-1]) & (self.robot.fitness != np.inf))

        # Selección de la mejor solución
        self.best_individual_index = np.argwhere(self.robot.fitness == self.max_fitness[-1])
        np.save("best_ind.npy", self.mp.solutions[self.best_individual_index[0][0]][self.best_individual_index[0][1]][0])

    def main(self):
        if self.steps == 0:
            if self.iteration < self.config.iterations and self.map:
                self.evolution()
            else:
                self.robot.current_weights = self.robot.weights[self.index]
                self.robot.nn.update_weights(self.robot.current_weights)
            
            if self.visualize:
                self.robot.nn.update_weights(self.mp.solutions[self.best_index[self.visual_index][0]][self.best_index[self.visual_index][1]][0])

        self.environment.simulate_step(self.steps)

        self.steps += 1 

        if self.steps == self.config.evaluation_steps or self.robot.obst_col:
            self.robot.x_final = self.robot.current_x
            self.robot.y_final =  self.robot.current_y
            self.environment.trail = []

            if self.visualize == False:
                self.evaluation()
            else:
                self.visual_index += 1

                if self.visual_index == len(self.best_index):
                    self.finished = True

            self.steps = 0
            self.environment.t = 0
            self.robot.reset()

            if self.map == False:
                self.index  += 1
            else:
                self.iteration += 1
                print("Iteración:" + "" +   str(self.iteration) + '/' + str(self.config.iterations))

            # Recorrer población inicial
            if self.index == self.config.population_size:
                self.map = True
                self.index = 0

            # Condición final evolución
            if self.iteration == self.config.iterations:
                self.select_best()
                self.mp.ploteo()
                self.visualize = True