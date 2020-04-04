import numpy as np
import vispy.scene
from vispy.scene import visuals
import matplotlib.pyplot as plt
import random

############################################FUNCTIONS###########################################
#2D RANDOM WALK
def randomwalk_step(current_position, current_velocity, dt): #this function returns a new position
    #we need  new velocity for this particle
    new_velocity = np.zeros(2)

    #change the x-velocity randomly
    new_velocity[0] = current_velocity[0] + 0.2*(np.random.rand()-0.5)
    new_velocity[0] = np.sign(new_velocity[0])*np.min([1, np.abs(new_velocity[0])]) #bound velocity

    #change the y-velocity randomly
    new_velocity[1] = current_velocity[1] + 0.2*(np.random.rand() - 0.5)
    new_velocity[1] = np.sign(new_velocity[1]) * np.min([1, np.abs(new_velocity[1])]) #bound velocity

    return new_velocity

#DISTANCE COMPUTATIONS
def compute_distances(current_pos, positions):
    return np.linalg.norm(positions-current_pos, ord=2, axis=1)

#####################################################SIMULATION PARAMETERS###############################################
#domain and timestepping
L = 1 #domain length and domain height
L_repel = 0.05 #this is the band in which particles are repelled from the boundary
N = 200 #population size
N_infected = 1 #number of initially infected people (should be smaller than N)
N_clean = N-N_infected #number of people that did not get the virus yet
N_immune = 0 #number of people that are immune
dt = 0.01 #time-step
timestep = 0 #number of of timesteps already taken
timestep_end = 1000 #number of timesteps to take
#velocity
velocity_scaling = 1 #this is the factor which we multiply with the velocity [-1, 1]
#infection
infection_radius = 0.05 #people might get infected when they come within this distance of an infected particle
infection_probability = 0.1 #change of getting infected when within the radius
infection_time = 0.5 #time a person will be infected
#social distancing
social_distance_radius = 0.1 #the radius in which the social distancing kicks in
social_distance_probability = 1 #the probability that a person obeys the social distancing law
social_distancing_percentage = 1 #fraction of people that obeys the social distancing law
#isolation
isolation_probability = 0 #fraction of infected that gets isolated
time_before_isolation = 0.2 # time a person walks around before being isolated
#supermarket
time_to_go = 0 # how often someone goes to supermarket
supermarket_percentage = 0 # fraction of people that goes to supermarket

####################################INITIALISE PARTICLES############################################################
zero_vector = np.zeros([N, 1]) #this is a vector that is used for plotting the points in a plane in 3D
particle_pos = L*np.random.rand(N, 2) #positions
particle_vel = 2*(np.random.rand(N, 2)-0.5) #movement velocity
#infection related
particle_infected = np.zeros(N, dtype=np.int) #indicates if a particle is infected or not (0=not infected, 1=infected, 2=immune)
particle_infectiontime = np.zeros(N) #indicates the time of when the particle was infected
#shopping related
shop_pos = np.random.uniform(L/4,3*L/4,2)
particle_shopping = np.zeros(N) #indicates the time of when the particle went shopping last
#social distancing
particle_social_distancing = np.zeros(N) #array which indicates if a particle does social distancing (0=no, 1=yes)
particle_social_distancing[random.sample(range(0, N), np.int(np.round(social_distancing_percentage*N)))] = 1
#infect first particles
for i in range(0, N_infected):
    index = np.random.randint(0, N) #pull a random integer (this represents the index of the infected particle
    particle_infected[index] = 1 #this particle is infected
particle_isolated = np.zeros(N) #array which indicates if a particle is isolated or not (0=not infected, 1=infected and will be in isolation, 2=infected but will not be isolated, 3=in isolation, 4=out of isolation)

##############################################PLOTTING################################################################
plotcounter = 1 #plot every "plotcounter" frames
color_scheme = np.array([[0, 0, 1, 0.5], [1, 0, 0, 0.5], [0, 1, 0, 0.5]]) #(clean, infected, immune) particle colours in RGBA format
#PARTICLE POSITIONS
canvas = vispy.scene.SceneCanvas(title="Particle positions", keys='interactive', show=True)
view = canvas.central_widget.add_view()
scatter = visuals.Markers()
scatter.set_data(np.concatenate((particle_pos, zero_vector), axis=1), edge_color='white', face_color=color_scheme[particle_infected], size=15)
view.add(scatter)
view.camera = 'panzoom'
# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)
#SIR GRAPH
plotSIR = 0 #(0=not show SIR graph, 1=show SIR graph)
fig, ax = plt.subplots(1, 1)
xSIR = np.arange(0, timestep_end+1, 1) #[0, 1, 2, 3, 4, 5, 6, ...]
yS = np.zeros(timestep_end+1) #number of susceptible people
yI = yS*1 #number of infected people over time
yR = yS*1 #number of immune people over time
yS[0] = N_clean
yI[0] = N_infected
yR[0] = N_immune

def update(event):
    global timestep, particle_infected, particle_pos, particle_social_distancing, particle_isolated, particle_vel, particle_infectiontime, particle_shopping, N_infected, N_clean, N_immune #global variables that need to be altered every timestep

    # NEW POSITIONS
    for i in range(0, N):  # loop over all particles to change position
        if particle_isolated[i]!=3: #check if particle is not in isolation
            current_pos = particle_pos[i, :]  # position of the particle
            current_vel = particle_vel[i, :]  # velocity of the particle
            repulsion_force = np.zeros(2)  # this is the repulsion force

            # new smooth velocity
            new_vel = randomwalk_step(current_pos, current_vel, dt)
            # force toward a central position (supermarket)
            if timestep * dt - particle_shopping[i] > time_to_go:  # if enough time has passed since last time he went shopping
                if np.random.rand() < supermarket_percentage:  # if this particle is indeed going shopping this time
                    vector = shop_pos - current_pos
                    new_vel = vector  # if its time to go shopping we move targeted towards shop
            if np.linalg.norm(current_pos - shop_pos, ord=2) < 0.01:  # if he has visited the shop approximately
                particle_shopping[i] = timestep * dt

            # repulsive force (social distancing)
            if social_distance_radius > 0:
                if particle_social_distancing[i] == 1:  # check if this particle obeys the social distancing physics
                    if np.random.rand() < social_distance_probability:  # check if he follows the law this time
                        # his position change would be dt*new_vel, but we need to make sure that this does not bring him closer to any of the other particles
                        particle_dist = compute_distances(current_pos,
                                                          particle_pos)  # compute the distance between the current particle and the remaining particles
                        particle_dist[particle_dist == 0] = L  # set the distance between itself to a large value
                        close_particles_indices = np.where(particle_dist < social_distance_radius)[0]
                        for j in close_particles_indices:  # loop over all close particles to add a force
                            vector = particle_pos[j] - current_pos  # vector of the repulsive force
                            norm = social_distance_radius * particle_dist[j]  # this is the distance from the particle
                            repulsion_force -= vector / (norm ** 3)  # why this force?
                        if np.linalg.norm(repulsion_force, ord=2) > 0:
                            new_vel += repulsion_force / np.linalg.norm(repulsion_force, ord=2)

            # wall force (keep particles away from the boundaries of the domain)
            if current_pos[0] < L_repel:
                new_vel[
                    0] = 1 * np.random.rand()  # the particle is too close to the left boundary and needs to go in the x-direction
            if current_pos[1] < L_repel:
                new_vel[
                    1] = 1 * np.random.rand()  # the particle is too close to the left boundary and needs to go in the y-direction

            # check if a particle is close to the top and/or right domain boundary
            if (L - current_pos[0]) < L_repel:
                new_vel[
                    0] = -1 * np.random.rand()  # the particle is too close to the left boundary and needs to go in the negative x-direction
            if (L - current_pos[1]) < L_repel:
                new_vel[
                    1] = -1 * np.random.rand()  # the particle is too close to the left boundary and needs to go in the negative y-direction

            particle_vel[i, :] = velocity_scaling * new_vel #scale the velocity with a maximum velocity

    # update particle positions
    particle_pos += dt * particle_vel
    # keep particles inside the domain
    particle_pos[particle_isolated!=3] = np.clip(particle_pos[particle_isolated!=3], [0, 0], [L, L])

    # NEW INFECTIONS
    infected_particles = np.where(particle_infected == 1)[0]  # currently infected people
    for i in range(0, np.size(infected_particles)): #loop over all infected particles
        infected_pos = particle_pos[infected_particles[i], :]  # this is the current position of the infected particle
        in_danger_indices = np.where(np.linalg.norm(particle_pos - infected_pos, ord=2, axis=1) < infection_radius)[0]  # find all the particles that are within the infection radius
        in_danger_indices = in_danger_indices[np.where(particle_infected[in_danger_indices] == 0)[0]]  # remove the indices of particles that are immune or are already infected
        new_infected_indices = in_danger_indices[np.random.rand(np.size(in_danger_indices)) < infection_probability]  # these are the indices of particles that are infected
        particle_infected[new_infected_indices] = 1  # set these particles to infected
        particle_infectiontime[new_infected_indices] = timestep * dt  # set the infection starting time to the current timelevel
        if isolation_probability>0:
            particle_isolated[new_infected_indices] = np.random.choice([1, 2], np.size(new_infected_indices), p=[isolation_probability, 1-isolation_probability]) #check if particle will get isolated or not
        N_infected += np.size(new_infected_indices)  # update the number of infected people
        N_clean -= np.size(new_infected_indices)  # update the number of clean people

    # NEW IMMUNE PEOPLE
    new_immune_indices = infected_particles[np.where(timestep * dt - particle_infectiontime[infected_particles] > infection_time)[0]]  # these are the infected people that are infected for a while
    particle_infected[new_immune_indices] = 2  # set these particles to immune
    N_infected -= np.size(new_immune_indices)  # update the number of infected people
    N_immune += np.size(new_immune_indices)  # update the number of infected people

    #PATICLES MOVING TO ISOLATION
    if isolation_probability>0:
        new_isolations = np.where(particle_isolated == 1)[0]  # particles that eventually need to be isolated
        new_isolations = new_isolations[np.where(timestep * dt - particle_infectiontime[new_isolations] > time_before_isolation)[0]] #check if the particle needs to be isolated now
        particle_isolated[new_isolations] = 3 #set the status of the particle to "in isolation"
        particle_pos[new_isolations] = [-0.3, -0.3] #move the particles to quarantine
        particle_vel[new_isolations] = [0, 0] #set them to be stationary

    # PATICLES GOING OUT ISOLATION
    if isolation_probability > 0:
        new_isolations = np.where(particle_isolated == 3)[0]  # particles that are in isolation now
        new_isolations = new_isolations[np.where(timestep * dt - particle_infectiontime[new_isolations] > infection_time)[0]]  # check if the particle needs to be isolated now
        particle_isolated[new_isolations] = 4  # set the status of the particle to "in isolation"
        particle_pos[new_isolations] = [L/2, L/2]  # move the particles to quarantine
        particle_vel[new_isolations] = velocity_scaling*2*(np.random.rand(2)-0.5)  # set them to be stationary

    #PLOT NEW PARTICLE POSITIONS
    scatter.set_data(np.concatenate((particle_pos, zero_vector), axis=1), edge_color='white',face_color=color_scheme[particle_infected], size=15)
    canvas.title = "Particle positions at t="+str(timestep*dt)

    #PLOT SIR GRAPH
    if plotSIR==1:
        yS[timestep] = N_clean
        yI[timestep] = N_infected
        yR[timestep] = N_immune
        ax.cla()
        Iplot = ax.fill_between(xSIR, 0, yI, color=color_scheme[1])
        Splot = ax.fill_between(xSIR, yI, yI + yS, color=color_scheme[0])
        Rplot = ax.fill_between(xSIR, yI + yS, yI + yS + yR, color=color_scheme[2])
        ax.axis([0, timestep, 0, N + 1])
        plt.draw()
        plt.pause(0.0000001)

    #UPDATE TIME
    timestep = timestep+1

timer = vispy.app.Timer('auto', connect=update, start=True)

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()