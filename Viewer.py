import sys
import math
import time
import pygame
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Environment import Environment
from NeuralNetwork import JordanNetwork
from Configuration import Configuration
from MAPELitesSimulation import MAPElitesSimulation
from DifferentialSimulation import DifferentialSimulation


BLACK = (0, 0, 0)
RED = (255,0,0)
GREY = (206, 204, 202)
WHITE = (255, 255, 255)
GREEN = (0, 100, 0)
BLUE = (0, 0, 255)

cm_pixel = 0.3

# Constantes movimiento lineal
a = 0.0000175535946480465
b = 0.379742197503453
c = 2.12361958525124
a_positiva = 103.560231228394
t0 = 0.1

class Robot:
    def __init__(self):
        # Inicialización
        self.current_x = 50
        self.current_y = 450
        self.sum_v_l = 0
        self.sum_v_r = 0
        self.v_current_r = 0
        self.v_current_l = 0
        self.obst_col = False
        self.x_inicial = self.current_x
        self.y_inicial = self.current_y

        # Inicialización visual
        self.x_visual = self.current_x
        self.y_visual = self.current_y
        self.alpha_visual = -math.pi/2

        self.fitness = []
        self.inputs = [0]*4
        self.outputs = [0]*2
        self.sum_vel = 0
        self.buffer_v = [[0,0]] * int(0.1/config.dt)
        self.buffer_sensor = [[0, 0, 0, 0]] * int(0.4/config.dt)

        # Inicialización red neuronal
        self.nn = JordanNetwork(4, 4, 2)

    def modelo_vel(self, v_percentage):

        if v_percentage == 0:
            v = 0
        else:
            v = math.pow(v_percentage, 2)*a + v_percentage*b + c
        return v

    def aceleracion(self, v_objetivo, v):

        a_negativa = v/0.208

        if v < v_objetivo:
            v += a_positiva*dt
        elif v > v_objetivo:
            v -= a_negativa*dt
            if v < 3:
                v = 0
        return v

    def act(self, v_r, v_l):

        # Cálculo de la nueva velocidad
        v_array = [self.aceleracion(self.modelo_vel(v_r*100*0.6), self.v_current_r), self.aceleracion(self.modelo_vel(v_l*100*0.6),  self.v_current_l)]
        self.v = (v_array[1] + v_array[0])/2
        self.prueba_array = v_array
        inc_theta = (v_array[1] - v_array[0])/14.5

        # Actualización posición robot
        self.alpha_visual += inc_theta
        self.current_x += self.v/cm_pixel*math.cos(-self.alpha_visual)*dt
        self.current_y -= self.v/cm_pixel*math.sin(-self.alpha_visual)*dt

        # Límites
        if self.current_x < 0 : self.current_x = 0
        if self.current_y < 0: self.current_y = 0
        if self.current_x > config.width - 30 : self.current_x = config.width - 30
        if self.current_y > config.height - 30: self.current_y = config.height - 30

        self.v_current_r = v_array[0]
        self.v_current_l = v_array[1]

    def check_collision(self):
        # Detección pared
        if any(x > 0.9 for x in self.inputs):
            self.obst_col = True
        
        # Detección obstáculo
        if pygame.sprite.spritecollide(robobo_sprite, obstacles_list, False):
            self.obst_col = True
    
    def reset(self):
        # Inicialización
        self.current_x = 50
        self.current_y = 450
        self.sum_v_l = 0
        self.sum_v_r = 0
        self.v_current_r = 0
        self.v_current_l = 0
        self.obst_col = False
        self.x_inicial = self.current_x
        self.y_inicial = self.current_y
        self.x_final = 0
        self.y_final = 0

        # Inicialización visual
        self.x_visual = self.current_x
        self.y_visual = self.current_y
        self.alpha_visual = -math.pi/2

        self.sum_vel = 0
        self.outputs = [0]*2
        self.buffer_v = [[0,0]] * int(0.1/config.dt)
        self.buffer_sensor = [[0, 0, 0, 0]] * int(0.4/config.dt)

        # Inicialización red neuronal
        self.nn = JordanNetwork(4, 4, 2)

class RoboboSprite(pygame.sprite.Sprite):
    def __init__(self, robobo):
        super().__init__()
        self.robobo = robobo
        self.image_orig = pygame.image.load("/home/sium/Dropbox/Universidad/TFG/scripts/simulador_3/robobo_small.png").convert_alpha()
        self.image_orig = pygame.transform.scale(self.image_orig, (60, 60))
        self.image = self.image_orig.copy()
        self.rect = self.image.get_rect()
        self.rot = -math.degrees(self.robobo.alpha_visual)
        
    def update(self):
        self.rot = -math.degrees(self.robobo.alpha_visual)
        new_image = pygame.transform.rotozoom(self.image_orig, self.rot, 1.0)

        self.image = new_image
        self.rect = self.image.get_rect()
        self.rect.center = (self.robobo.x_visual, self.robobo.y_visual)

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, color, width, height, x_coordenate, y_coordenate, move_obstacles):
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.image.fill(WHITE)
        self.image.set_colorkey(WHITE)
        self.name = "circle"
        self.color = color
        self.radius = width//2
        self.center = [self.radius, self.radius]
        self.rect = self.image.get_rect()
        pygame.draw.circle(self.image, self.color, self.center, self.radius)
        self.rect.center = (x_coordenate, y_coordenate)
        self.move_obstacles = move_obstacles
        self.x_origin = x_coordenate
        self.y_origin = y_coordenate
        self.angle = random.uniform(0, math.pi)
        self.paso = math.pi/200

    def update(self):
        if self.move_obstacles:
            x_coordenate = self.x_origin + np.cos(self.angle)*40
            y_coordenate = self.y_origin + np.sin(self.angle*2)*40

            self.angle += self.paso
            self.rect.center = (x_coordenate, y_coordenate)

class circle(pygame.sprite.Sprite):
    def __init__(self, color, width, height, obstacle, alpha):
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.image.fill(WHITE)
        self.image.set_colorkey(WHITE)
        self.image.set_alpha(alpha)
        self.name = "circle"
        self.color = color
        self.radius = width//2
        self.center = [self.radius, self.radius]
        self.rect = self.image.get_rect()
        pygame.draw.circle(self.image, self.color, self.center, self.radius)
        self.obstacle = obstacle
        self.rect.center = self.obstacle.rect.center
    
    def update(self):
        self.rect.center = self.obstacle.rect.center

class rectangle(pygame.sprite.Sprite):
    def __init__(self, color, alpha, size):
        super().__init__()

        self.image = pygame.Surface([config.width, config.height])
        self.image.fill(WHITE)
        self.image.set_colorkey(WHITE)
        self.image.set_alpha(alpha)
        self.name = "rectangle"
        self.color = color
        self.rect = self.image.get_rect()
        pygame.draw.rect(self.image, self.color, (0, 0, size, config.height),0)
        pygame.draw.rect(self.image, self.color, (0, 0, config.width, size),0)
        pygame.draw.rect(self.image, self.color, (config.width - size, 0, size, config.height),0)
        pygame.draw.rect(self.image, self.color, (0, config.height - size, config.width, size),0)
        

config = Configuration()
dt = config.dt

# Configuración pantalla
display = pygame.display.set_mode((config.width, config.height))
pygame.display.set_caption("Simulation")
pygame.font.init() 
font = pygame.font.SysFont('arial', 20)
clock = pygame.time.Clock()
pygame.init()

# Background
bg = pygame.image.load("/home/sium/Dropbox/Universidad/TFG/scripts/simulador_3/bg.png")

# Inicialización sprites
obstacle_1 = Obstacle(GREEN, 30, 30, 350, 350, config.move_obstacles)
obstacle_2 = Obstacle(GREEN, 30, 30, 150, 350, config.move_obstacles)
obstacle_3 = Obstacle(GREEN, 30, 30, 350, 150, config.move_obstacles)
obstacle_4 = Obstacle(GREEN, 30, 30, 150, 150, config.move_obstacles)

# Zona frenado
zone_1 = circle(RED, 115, 115, obstacle_1, 50)
zone_2 = circle(RED, 115, 115, obstacle_2, 50)
zone_3 = circle(RED, 115, 115, obstacle_3, 50)
zone_4 = circle(RED, 115, 115, obstacle_4, 50)

zone_5 = circle(RED, 80, 80, obstacle_1, 125)
zone_6 = circle(RED, 80, 80, obstacle_2, 125)
zone_7 = circle(RED, 80, 80, obstacle_3, 125)
zone_8 = circle(RED, 80, 80, obstacle_4, 125)

rect_zone_1 = rectangle(RED, 50, 50)
rect_zone_2 = rectangle(RED, 125, 25)

obstacles_list = pygame.sprite.Group()
obstacles_list.add(obstacle_1)
obstacles_list.add(obstacle_2)
obstacles_list.add(obstacle_3)
obstacles_list.add(obstacle_4)

#obstacles_list = []

zone_list = pygame.sprite.Group()

zone_list.add(zone_1)
zone_list.add(zone_2)
zone_list.add(zone_3)
zone_list.add(zone_4)
zone_list.add(zone_5)
zone_list.add(zone_6)
zone_list.add(zone_7)
zone_list.add(zone_8)
zone_list.add(rect_zone_1)
zone_list.add(rect_zone_2)

robot = Robot()
robobo_sprite = RoboboSprite(robot)

sprites_list = pygame.sprite.Group()
sprites_list.add(robobo_sprite)

Environment = Environment(robot, config, obstacles_list)
algorithm = config.algorithm
if algorithm == 0:
    sim = MAPElitesSimulation(config, robot, Environment)
elif algorithm == 1:
    sim = DifferentialSimulation(config, robot, Environment)

running = True

t_1 = int(round(time.time() * 1000))

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Evolución
    sim.main()

    sprites_list.update()
    obstacles_list.update()
    zone_list.update()

    if sim.finished:
        running = False

    if sim.visualize:
        display.fill(WHITE)
        display.blit(bg, (0, 0))

        # Representar elementos del entorno 
        zone_list.draw(display)
        for cord in range(len(sim.environment.trail)):
            pygame.draw.circle(display, BLUE, sim.environment.trail[cord], 4)
        sprites_list.draw(display)
        obstacles_list.draw(display)

        iteration_str = 'Iteration: ' + str(sim.iteration + 1) + ' ' + 'Velocidad: ' + str(np.round(robot.v, 2))
        text = font.render(iteration_str, True, WHITE)
        display.blit(text, [10, 10])
        pygame.display.flip()
        clock.tick(config.fps)
pygame.quit()

print(int(round(time.time() * 1000))- t_1)

sns.set()
sns.set_style("whitegrid")
x = np.arange(len(sim.mean_fitness))

if algorithm == 1:
    plt.fill_between(x, np.array(sim.mean_fitness) - np.array(sim.deviation_fitness), np.array(sim.mean_fitness) + np.array(sim.deviation_fitness), color='b', alpha=0.2)
    plt.plot(x, sim.mean_fitness, '-b', label="calidad media", markevery = 500, marker='o')
    plt.plot(x, sim.max_fitness, '-r', label="calidad máxima", markevery = 500, marker='s')
    plt.legend(loc="lower right")
    plt.ylabel('Calidad')
    plt.xlabel('Evaluaciones')
    plt.xlim(0, len(x))
    plt.ylim(0,1)
    plt.savefig('calidad_mean_max')
    plt.clf()

    for i in range(sim.fitnes_matrix.shape[0]):
        plt.plot(sim.fitnes_matrix[i][:], '-.')
    plt.ylabel('Calidad')
    plt.xlabel('Generaciones')
    plt.xlim(0, config.iterations - 1)
    plt.ylim(0,1)
    plt.savefig('calidad_pob')
    plt.clf()

    for i in range(sim.fitnes_matrix.shape[0]):
        plt.plot(sim.gen_matrix[i][:], '-.')
    plt.ylabel('Gen Medio')
    plt.xlabel('Generaciones')
    plt.xlim(0, config.iterations - 1)
    plt.ylim(0,1)
    plt.savefig('gen_mean')
    plt.clf()

else:

    plt.fill_between(x, np.array(sim.mean_fitness) - np.array(sim.deviation_fitness), np.array(sim.mean_fitness) + np.array(sim.deviation_fitness), color='b', alpha=0.2)
    plt.plot(x, sim.mean_fitness, '-b', label="calidad media", markevery = 500, marker='o')
    plt.plot(x, sim.max_fitness, '-r', label="calidad máxima", markevery = 500, marker='s')
    plt.legend(loc="lower right")
    plt.ylabel('Calidad')
    plt.xlabel('Evaluaciones')
    plt.xlim(0,config.iterations - 1)
    plt.ylim(0,1)
    plt.savefig('map_calidad')