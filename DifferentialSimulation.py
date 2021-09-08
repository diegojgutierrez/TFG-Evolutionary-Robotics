import diff
import numpy as np

class DifferentialSimulation:
    def __init__(self, config, robot, environment):
        self.index = 0
        self.steps = 0
        self.iteration = 0
        self.finished = False
        self.visualize = False
        self.robot = robot
        self.config = config
        self.environment = environment
        self.dimensions = self.robot.nn.size
        self.robot.weights = diff.generate_initial_population(config.low, config.high, self.dimensions, config.population_size)
        self.current_weights = self.robot.weights[0]

        self.max_fitness = []
        self.mean_fitness = []
        self.deviation_fitness = []
        self.fitnes_matrix = np.full((self.config.population_size, self.config.iterations),np.inf)
        self.gen_matrix = np.full((self.config.population_size, self.config.iterations),np.inf)

    def evolution(self):

        # Mutación 
        noisy_vector = diff.mutation(0, 1, self.robot.weights, self.config.population_size, self.config.F)
        self.robot.current_weights = diff.recombination(noisy_vector, self.robot.weights[self.index], self.dimensions, self.config.cross_probability)

        # Actualización pesos de la red
        self.robot.nn.update_weights(self.robot.current_weights)

    def calculate_fitness(self):

        v_mean = (self.robot.sum_vel/self.steps)
        v_diff = 1 - np.sqrt(np.absolute(self.robot.sum_v_l/(self.steps + 1) - self.robot.sum_v_r/(self.steps + 1)))
        fitness = v_mean * v_diff * self.steps/self.config.evaluation_steps

        return  fitness

    def evaluation(self):
            
        # Evaluación
        if self.iteration > 0:
            # Maximización
            if self.calculate_fitness() > self.robot.fitness[self.index]:
                self.robot.fitness[self.index] = self.calculate_fitness()
                self.robot.weights[self.index] = self.robot.current_weights
        else:
            self.robot.fitness.append(self.calculate_fitness())
            self.robot.weights[self.index] = self.robot.current_weights

        #print(self.calculate_fitness())

        if self.iteration < self.config.iterations:
            self.fitnes_matrix[self.index][self.iteration] = self.robot.fitness[self.index]
            self.gen_matrix[self.index][self.iteration] = sum(self.robot.weights[self.index])/self.dimensions
            self.mean_fitness.append(np.mean(self.robot.fitness))
            self.max_fitness.append(max(self.robot.fitness))
            self.deviation_fitness.append(np.std(self.robot.fitness))

    def select_best(self):
        # Selección de la mejor solución
        self.best_individual_index = np.argwhere(self.robot.fitness == self.max_fitness[-1])
        np.save("best_ind.npy", self.robot.weights[self.best_individual_index[0][0]])
        
    def main(self):
        if self.steps == 0:
            if self.iteration < self.config.iterations:
                self.evolution()
            else:
                self.robot.nn.update_weights(self.robot.weights[self.index])

        self.environment.simulate_step(self.steps)
        self.steps += 1
        
        if self.steps == self.config.evaluation_steps or self.robot.obst_col:
            self.robot.x_final = self.robot.current_x
            self.robot.y_final =  self.robot.current_y
            self.environment.trail = []

            if self.visualize == False:
                self.evaluation()
            
            self.steps = 0
            self.environment.t = 0
            self.robot.reset()

            self.index += 1

            # Recorrer miembros población
            if self.index == self.config.population_size:
                self.iteration += 1
                self.index = 0
                print("Iteración:" + "" +   str(self.iteration) + '/' + str(self.config.iterations))

            # Condición final evolución
            if self.iteration == self.config.iterations:
                self.select_best()
                self.visualize = True   
            
            # Condición final evolución
            if self.iteration == self.config.iterations + 1:
                self.finished = True