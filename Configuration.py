class Configuration():
    def __init__(self):
        # Parámetros Simulador
        # Original 500 x 500
        self.width = 500
        self.height = 500
        self.fps = 60
        self.dt = 1/self.fps
        self.evaluation_steps = 700
        self.move_obstacles = False
        self.algorithm = 0

        # Parámetros EAs Generales
        self.population_size = 1000
        self.iterations = 10000

        # Parámetros Differential Evolution
        self.low = 0
        self.high = 1
        self.F = 0.1
        self.cross_probability = 0.9

        # Parámetros MAP-Elites

        # Feature Space
        self.upperlimit = self.height
        self.lowerlimit = 0
        self.feature_dimensions = 2
        self.granulation = 10

        # Parámetros Evolución
        self.crossover_rate = 0.2
        self.mutation_rate = 0.8
        self.sigma = 0.05
        self.cross = False