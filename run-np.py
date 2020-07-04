import pygame
import random
import numpy as np



class Mass(object):
    STAR_BRIGHT = (255, 240, 220)
    SPACE_BLACK = (0, 0, 15)
    DTYPE = 'int16'



    def __init__(self, start_x, start_y, num_particles=3, mass=1, damping_coeffiecnt=1):
        # TODO: Add an option to import 2D models later
        self.define_mass(start_x, start_y, num_particles, mass, damping_coeffiecnt)



    def define_mass(self, start_x, start_y, num_particles, mass, damping_coeffiecnt):
        # Defining the particles cluster variables
        start_position = np.array([[start_x], [start_y]], dtype=self.DTYPE)
        self.size = num_particles
        self.mass = mass

        self.mass_vector = np.full([1, num_particles], mass, dtype=self.DTYPE)
        self.damping_vector = np.full([1, num_particles], damping_coeffiecnt, dtype=self.DTYPE)
        self.gravity = np.array([[0],[0]])

        self.position = np.array([[1,0,3], [-1, 2, 5]])
        self.velocity = np.zeros([2, num_particles])
        self.stiffness_matrix = np.array([[0,1,0],[1,0,9],[0,9,0]])
        self.free_length = np.array([[0, 0.5, 0],[0.5,0,7],[0,7,0]])



    def calculate_next_position(self, time_step):
        for i in range(self.size): 
            d_pos = self.position - self.position[:,i:i+1]
            norm = np.divide(d_pos, np.linalg.norm(d_pos, axis = 0))
            norm = np.nan_to_num(norm)
            FLWD = np.multiply(norm ,self.free_length[:,i]) # a zal
            AAF = np.subtract(d_pos, FLWD) # b zal 
            elastic_force = AAF @ self.stiffness_matrix[:, i:i+1]
            damping_force = np.multiply(self.damping_vector[i], self.velocity[:,i:i+1])
            gravity_force = np.multiply(self.mass_vector[i], self.gravity)
            total_force = np.add(elastic_force , damping_force , gravity_force)
            # runge kutta method solver

            new_pos[:, i:i+1] = position[:, i:i+1] + (velocity[:, i:i+1] * time_step)
            new_vel[:, i:i+1] = velocity[:, i:i+1] + (total_force/mass_vector[i] * time_step)
        self.position = new_pos
        self.velocity = new_vel


    def get_pos(self):
        return self.position






class Particle(object):
    DTYPE = 'int8'
    elastic_coefiecnt = 2
    air_coefiecnt = 10
    gravity_down = 0



    def __init__(self, start_x=0, start_y=0, mass=1):
        # Defining base variables
        self.mass = mass 
        self.position = np.array([start_x, start_y], dtype=self.DTYPE)
        self.velocity = np.zeros((2), dtype=self.DTYPE)



    def calc_next_position(self, pull_position, time_step = 0.1):
        # Calculating next position using Euler's method
        elastic_force = np.array((2), dtype=self.DTYPE)
        friction_force = np.array((2), dtype=self.DTYPE)
        gravity = np.array([0, self.gravity_down], dtype=self.DTYPE)

        elastic_force = np.multiply(np.subtract(pull_position, self.position), self.elastic_coefiecnt/self.mass)
        friction_force = np.multiply(self.velocity, self.air_coefiecnt/self.mass)
        el_n_f = np.subtract(elastic_force, friction_force)
        total_force = np.add(el_n_f, gravity)
        self.velocity = np.add(self.velocity, np.multiply(total_force, time_step))
        self.position = np.add(self.position, self.velocity)



    def get_position_int(self):
        # Return the position of the particle as an integer
        return int(self.position[0]), int(self.position[1])










class SimO(object):
    WINSIZE = [1280, 960]
    WINCENTER = [640, 480]
    SPACE_BLACK = (0, 0, 15)
    STAR_BRIGHT = (255, 230, 200)
    FPS = 60
 
 

    def __init__(self):
        random.seed()
        pygame.init()
        pygame.display.set_caption("Sim")
        self.clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode(self.WINSIZE, pygame.DOUBLEBUF)
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill(self.SPACE_BLACK)

        self.sim_time = 0.0
        self.font = pygame.font.SysFont('mono', 20)

        self.particles = []
        self.particles_color = []

        mass = []
        # Create an array of particles
        for i in range(750):
            self.particles.append(Particle(random.randint(0,1280), random.randint(0,960), random.randint(5,1000)))
            
            # Find maximum and minmum mass values for each particle (because its random ..)
            mass.append(self.particles[i].mass)
            
        # Calculate the color based on each particle mass (More mass = more bright) 
        self.particles_color = np.clip(mass, 150, 255) # (min brightness = 150 -- max brightness = 255)



    def run(self):
        mouse_pos = [640, 480]
        done = False

        while not done:
            ms = self.clock.tick(self.FPS)
            time_step = ms / 1000.0
            self.sim_time += ms / 1000.0


            for color, particle in zip(self.particles_color, self.particles):
                self.draw_pixel(self.background, particle.get_position_int(), self.SPACE_BLACK)

                particle.calc_next_position(mouse_pos, time_step)
                self.draw_pixel(self.background, particle.get_position_int(), (color, color, color))
                
            # Update screen
            self.screen.blit(self.background, (0,0))
            
            self.draw_text("Simulation Time: {0:.3f}".format(self.sim_time),[640, 50])
            self.draw_text("Time step: {0:.3f}".format(time_step),[640, 100])
            self.draw_text("FPS: {0:.3f}".format(self.clock.get_fps()),[640, 150])
            pygame.display.update()

            # Handle user interaction events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.type == pygame.K_ESCAPE:
                        done = True
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos[:] = list(event.pos)      



    def draw_pixel(self, surface, position, color):
        surface.set_at(position, color)


        
    def draw_text(self, text, position=WINCENTER):
        fw, fh = self.font.size(text)
        text_surface = self.font.render(text, True, self.STAR_BRIGHT)
        self.screen.blit(text_surface, (position[0] - (fw/2), position[1] - (fh/2)))










if __name__ == "__main__":
    SimO().run()
