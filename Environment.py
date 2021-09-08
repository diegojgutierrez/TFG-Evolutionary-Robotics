import math
import numpy as np

TWOPI = 2*np.pi

def normalize_angle(angle):
    # reduce the angle

    angle = angle % TWOPI
    # en python % es modulo entonces 0 <= angle < 360

    # force into the minimum absolute value residue class, so that -180 < angle <= 180
    if angle > np.pi:  
        angle -= TWOPI
    return angle

class Environment:
    def __init__(self, robot, config, obstacles):
        self.t = 0
        self.dt = config.dt
        self.robot = robot

        # parametros para sensor rendija
        self.alpha_max = math.radians(25)
        self.tam_rendija = 4
        self.H0 = 1
        self.w0 = self.tam_rendija
        self.width = config.width
        self.height = config.height
        self.obstacles = obstacles
        self.radio_robobo = 30
        self.trail = []

    def simulate_step(self, step):

        # Actualizar sensores robo
        self.actualizar_sensores_de_robot()
        self.robot.buffer_sensor.append(self.robot.inputs)
        normal_input = np.array(self.robot.buffer_sensor[step])

        #Red neuronal (Entrada: Rendija, Salida: Velocidad ruedas)
        self.robot.outputs = self.robot.nn.propagate_forward(normal_input)
    
        self.robot.buffer_v.append([self.robot.outputs[0] , self.robot.outputs[1]])

        self.robot.sum_vel += (self.robot.outputs[0] + self.robot.outputs[1])/2
        self.robot.sum_v_l += self.robot.outputs[1]
        self.robot.sum_v_r += self.robot.outputs[0]

        # Actuadores
        self.robot.act(self.robot.buffer_v[step][0],  self.robot.buffer_v[step][1])
        self.robot.check_collision()

        # Actualizar copia gráfica
        self.update_visual()

        self.t += self.dt

    def update_visual(self):
        self.robot.x_visual = self.robot.current_x
        self.robot.y_visual = self.robot.current_y
        self.trail.append((int(self.robot.x_visual),int(self.robot.y_visual)))

    def actualizar_sensores_de_robot(self):
        
        x_j = self.robot.current_x
        y_j = self.robot.current_y
        alpha_j = self.robot.alpha_visual
        self.D0 = self.radio_robobo
        self.D0_squared = self.D0*self.D0
        
        array_x_w = []

        for obstacle in self.obstacles:
            x_i = obstacle.rect.center[0]
            y_i = obstacle.rect.center[1]
            alfa_ij = math.atan2(y_i - y_j, x_i - x_j)
            alfa_rij = normalize_angle(alfa_ij - alpha_j)
            if abs(alfa_rij) <= self.alpha_max:
                x_ij = (alfa_rij + self.alpha_max)/(2*self.alpha_max)
                D_ij = (x_i - x_j)*(x_i - x_j) + (y_i - y_j)*(y_i - y_j)
                w_ij = self.w0 * self.D0_squared/D_ij
                array_x_w.append((x_ij, w_ij))
            
        alpha_ray_izq = normalize_angle(alpha_j - self.alpha_max)
        alpha_ray_der = normalize_angle(alpha_j + self.alpha_max)
            
        di_j = self.corte_pared_distancia(x_j, y_j, alpha_ray_izq)
        dd_j = self.corte_pared_distancia(x_j, y_j, alpha_ray_der)
        #print(di_j, dd_j)
                
        HI_j = self.H0*self.D0*2 / di_j
        HD_j = self.H0*self.D0*2 / dd_j
        #print(HI_j,HD_j)

        self.actualizar_rendija(self.robot, HI_j, HD_j, array_x_w)

    def detectar_corte_pared(self, x, y, robot, alpha, alpha_max):

        alpha1 = normalize_angle(alpha + alpha_max)
        dist_corte1 = self.corte_pared_distancia(x, y, alpha1)
        if dist_corte1 > 0 and dist_corte1 < self.robot.distance_closest:
            self.robor.distance_closest = dist_corte1
            self.robor.angle_closest = alpha1
        
        alpha2 = normalize_angle(alpha - alpha_max)
        dist_corte2 = self.corte_pared_distancia(x, y, alpha2)
        if dist_corte2 > 0 and dist_corte2 < self.robor.distance_closest:
            self.robor.distance_closest = dist_corte2
            self.robor.angle_closest = alpha2

    def actualizar_rendija(self, robot, HI_j, HD_j, array_xsws):
    
        # de i 0 a 4 (rendija)
        for i in range(self.tam_rendija):
            v = 0
            for j in range(len(array_xsws)):
                (x,w) = array_xsws[j]
                v += max(0,min(1,self.tam_rendija*(0.5/self.tam_rendija - (abs(x-(i+0.5)/self.tam_rendija) - w/2))))
            if v > 0.001:
                v = max(-v, -1)
            else:
                v = HI_j + i * (HD_j - HI_j) / (self.tam_rendija-1)
                if v > 1:
                    v = 1.0
            self.robot.inputs[i] = v

            # alpha_ray en rad
    def corte_pared_distancia(self, x_i, y_i, alpha_ray):

        xc = math.inf
        yc = math.inf
        dc = math.inf

        cos_alpha = math.cos(alpha_ray)
        sin_alpha = math.sin(alpha_ray)

        #print("Corte pared x", x_i, ", y", y_i, ", alpha_ray (º)", math.degrees(alpha_ray))
        (dN, xuN, yuN) = intersection_metodo_nuevo(x_i, y_i, alpha_ray, cos_alpha, sin_alpha, 0, 0, self.width, 0)
        if dN > 0 and dN < dc:
            xc = xuN
            yc = yuN
            dc = dN
        #print("Norte xc, yc, dc", xuN , yuN, dN)
        (dS, xuS, yuS) = intersection_metodo_nuevo(x_i, y_i, alpha_ray, cos_alpha, sin_alpha, 0, self.height, self.width, self.height)
        if dS > 0 and dS < dc:
            xc = xuS
            yc = yuS
            dc = dS
        #print("Sur xc, yc, dc", xuS, yuS, dS)
        (dE, xuE, yuE) = intersection_metodo_nuevo(x_i, y_i, alpha_ray, cos_alpha, sin_alpha, 0, 0, 0, self.height)
        if dE > 0 and dE < dc:
            xc = xuE
            yc = yuE
            dc = dE
        #print("Este xc, yc, dc", xuE, yuE, dE)
        (dO, xuO, yuO) = intersection_metodo_nuevo(x_i, y_i, alpha_ray, cos_alpha, sin_alpha, self.width, 0, self.width, self.height)
        if dO > 0 and dO < dc:
            xc = xuO
            yc = yuO
            dc = dO
        #print("Oeste xc, yc, dc", xuO, yuO, dO)
        #print("x corte, y corte, d", xc, yc, dc)
        return dc

    def crear_array_contodo(self, HI_j, HD_j, array_xsws):
    
        array = [0] * self.tam_rendija
        for i in range(self.tam_rendija):
            v = 0
            for ix in range(len(array_xsws)):
                x = array_xsws[ix][0]
                w = array_xsws[ix][1]
                v += max(0,min(1,1 - (abs(x-i) - w/2)))
                #print(1 - (abs(x-i) - w/2), abs(x-i))
                #print(x, w, v)
            if v > 0.001:
                v = max(-v, -1)
            else:
                v = HI_j + i * (HD_j - HI_j) / (self.tam_rendija-1)
            array[i] = v

        return array

    def combinar_pared_vecinos(self, arraypared, arrayvecinos):
        array = arraypared
        for i in range(len(arraypared)):
            if arrayvecinos[i] > 0.001:
                array[i] = -arrayvecinos[i]
                # cortar por -1
                if array[i] < -1:
                    array[i] = -1
            else:
                array[i] = arraypared[i]
                # cortar pr 1
                if array[i] > 1:
                    array[i] = 1
        return array

def intersection_metodo_nuevo(x1, y1, alpha, cos_alpha, sin_alpha, x2a, y2a, x2b, y2b):

    angulo_recta2 = math.atan2(y2b - y2a, x2b - x2a)
    dxa = cos_alpha
    dya = sin_alpha

    dxb = math.cos(angulo_recta2)
    dyb = math.sin(angulo_recta2)

    divisor = dxa*dyb - dxb*dya
    if divisor != 0:
        t = (dyb*(x2a-x1)-dxb*(y2a-y1))/divisor
        vx = dxa*t
        vy = dya*t

        # si d esta hacia atras signo negativo
        d = math.copysign(math.hypot(vx, vy), vx*cos_alpha + vy*sin_alpha)
        return (d, x1 + vx, y1 + vy)
    else:
        return (math.inf, math.nan, math.nan)