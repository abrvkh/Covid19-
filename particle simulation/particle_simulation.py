import numpy as np
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

#####################################################SIMULATION######################################################
##SIMULATION PARAMETERS
L = 1 #domain length and domain height
L_repel = 0.05 #this is the band in which particles are repelled from the boundary
N = 500 #population size
N_infected = 1 #number of initially infected people (should be smaller than N)
N_clean = N-N_infected #number of people that did not get the virus yet
N_immune = 0 #number of people that are immune
dt = 0.01 #time-step
timesteps = 100 #number of timesteps to take

##VELOCITY
velocity_scaling = 1 #this is the factor which we multiply with the velocity [-1, 1]

##INFECTION PARAMETERS
infection_radius = 0.05 #people might get infected when they come within this distance of an infected particle
infection_probability = 0.1 #change of getting infected when within the radius
infection_time = 0.14 #time a person will be infected

#SOCIAL DISTANCE PARAMETERS
social_distance_radius = 0.1 #the radius in which the social distancing kicks in
social_distance_probability = 1 #the probability that a person obeys the social distancing law
social_distancing_percentage = 1 #fraction of people that obeys the social distancing law

#TESTING PARAMETERS
test_percentage = 0 #fraction of people that gets tested and if positive he gets isolated
test_frequency = .1 # how frequent (in terms of nr of time steps) we test

# ISOLATION PARAMETERS
isolate_percentage = .1 #fraction of infected that gets isolated
isolate_time = 0 # time a person walks around before being isolated

# CENTRAL POINT (SUPERMARKET)
time_to_go = 0.15 # how often someone goes to supermarket
supermarket_percentage = .5 # fraction of people that goes to supermarket

#INITIALISE PARTICLES
particle_pos = L*np.random.rand(N, 2) #positions
particle_vel = 2*(np.random.rand(N, 2)-0.5) #movement velocity
#infection related
particle_infected = np.zeros(N, dtype=np.int) #indicates if a particle is infected or not (0=not infected, 1=infected, 2=immune)
particle_infectiontime = np.zeros(N) #indicates the time of when the particle was infected
# shopping related
shop_pos = np.random.uniform(L/4,3*L/4,2)
particle_shopping = np.zeros(N) #indicates the time of when the particle went shopping last
#social distancing
particle_social_distancing = np.zeros(N) #array which indicates if a particle does social distancing (0=no, 1=yes)
particle_social_distancing[random.sample(range(0, N), np.int(np.round(social_distancing_percentage*N)))] = 1

#INFECT FIRST PARTICLES
for i in range(0, N_infected):
    index = np.random.randint(0, N) #pull a random integer (this represents the index of the infected particle
    particle_infected[index] = 1 #this particle is infected

#PLOT PARTICLES
fig, (ax1, ax2) = plt.subplots(1, 2)
plotcounter = 1 #plot every "plotcounter" frames
#plot of the particles
color_scheme = np.array(['blue', 'red', 'green']) #(clean, infected, immune) particle colours
scatter_plot = ax1.scatter(particle_pos[:, 0], particle_pos[:, 1])
ax1.plot(shop_pos[0], shop_pos[1], 'mo')
scatter_plot.set_sizes(10*np.ones(np.size(particle_pos, 1)))
scatter_plot.set_facecolor(color_scheme[particle_infected])
scatter_plot.set_edgecolor(color_scheme[particle_infected])
ax1.axis([0, L, 0, L])
#plot of the SIR population
xSIR = np.arange(0, timesteps+1, 1) #[0, 1, 2, 3, 4, 5, 6, ...]
yS = np.zeros(timesteps+1) #number of susceptible people
yI = yS*1 #number of infected people over time
yR = yS*1 #number of immune people over time
yS[0] = N_clean
yI[0] = N_infected
yR[0] = N_immune

#TIME-LOOP
for it in range(1, timesteps+1): #time loop

    #NEW POSITIONS
    for i in range(0, N): #loop over all particles to change position
        current_pos = particle_pos[i, :] #position of the particle
        current_vel = particle_vel[i, :] #velocity of the particle
        repulsion_force = np.zeros(2) #this is the repulsion force

        #new smooth velocity
        new_vel = randomwalk_step(current_pos, current_vel, dt)
        # force toward a central position (supermarket)
        if it*dt - particle_shopping[i] > time_to_go: # if enough time has passed since last time he went shopping
            if np.random.rand()<supermarket_percentage: # if this particle is indeed going shopping this time
                vector = shop_pos-current_pos
                new_vel = vector # if its time to go shopping we move targeted towards shop
        if np.linalg.norm(current_pos-shop_pos, ord=2) < 0.01: # if he has visited the shop approximately
            particle_shopping[i] = it*dt

        #repulsive force (social distancing)
        if social_distance_radius>0:
            if particle_social_distancing[i]==1: #check if this particle obeys the social distancing physics
                if np.random.rand()<social_distance_probability: #check if he follows the law this time
                    # his position change would be dt*new_vel, but we need to make sure that this does not bring him closer to any of the other particles
                    particle_dist = compute_distances(current_pos, particle_pos) #compute the distance between the current particle and the remaining particles
                    particle_dist[particle_dist==0] = L #set the distance between itself to a large value
                    close_particles_indices = np.where(particle_dist<social_distance_radius)[0]
                    for j in close_particles_indices: #loop over all close particles to add a force
                        vector = particle_pos[j] - current_pos #vector of the repulsive force
                        norm = social_distance_radius*particle_dist[j] #this is the distance from the particle
                        repulsion_force -= vector/(norm**3) #why this force?
                    if np.linalg.norm(repulsion_force, ord=2)>0:
                        new_vel += repulsion_force/np.linalg.norm(repulsion_force, ord=2)

        #wall force (keep particles away from the boundaries of the domain)
        if current_pos[0] < L_repel:
            new_vel[0] = 1 * np.random.rand()  # the particle is too close to the left boundary and needs to go in the x-direction
        if current_pos[1] < L_repel:
            new_vel[1] = 1 * np.random.rand()  # the particle is too close to the left boundary and needs to go in the y-direction

        # check if a particle is close to the top and/or right domain boundary
        if (L - current_pos[0]) < L_repel:
            new_vel[0] = -1 * np.random.rand()  # the particle is too close to the left boundary and needs to go in the negative x-direction
        if (L - current_pos[1]) < L_repel:
            new_vel[1] = -1 * np.random.rand()  # the particle is too close to the left boundary and needs to go in the negative y-direction

        particle_vel[i, :] = velocity_scaling*new_vel

    #update particle positions
    particle_pos += dt*particle_vel

    #keep particles inside the domain
    particle_pos = np.clip(particle_pos, [0, 0], [L, L])

    #NEW INFECTIONS
    infected_indices = np.where(particle_infected == 1)[0] #currently infected people
    for i in range(0, np.size(infected_indices)):
        infected_pos = particle_pos[infected_indices[i], :] #this is the current position of the infected particle
        in_danger_indices = np.where(np.linalg.norm(particle_pos-infected_pos, ord=2, axis=1) < infection_radius)[0] #find all the particles that are within the infection radius
        in_danger_indices = in_danger_indices[np.where(particle_infected[in_danger_indices]==0)[0]] #remove the indices of particles that are immune or are already infected
        new_infected_indices = in_danger_indices[np.random.rand(np.size(in_danger_indices)) < infection_probability] #these are the indices of particles that are infected
        particle_infected[new_infected_indices] = 1 #set these particles to infected
        particle_infectiontime[new_infected_indices] = it*dt #set the infection time to the current timelevel
        N_infected += np.size(new_infected_indices) #update the number of infected people
        N_clean -= np.size(new_infected_indices) #update the number of clean people

    #NEW IMMUNE PEOPLE
    infected_particles = np.where(particle_infected==1)[0]
    new_immune_indices = infected_particles[np.where(it*dt - particle_infectiontime[infected_particles] > infection_time)[0]] #these are the infected people that are infected for a while
    particle_infected[new_immune_indices] = 2 #set these particles to immune
    N_infected -= np.size(new_immune_indices)  # update the number of infected people
    N_immune += np.size(new_immune_indices)  # update the number of infected people

    # ISOLATE POSITIVES
    infected_particles = np.where(particle_infected==1)[0]
    positives_tested = np.zeros(N_infected)
    positives_tested[:int(isolate_percentage*N_infected)] = 1
    np.random.shuffle(positives_tested)
    indices_tested = np.where(positives_tested==1)[0]
    indices_isolate = np.where(it*dt - particle_infectiontime[infected_particles]>isolate_time)[0]
    to_isolate = infected_particles[np.intersect1d(indices_tested,indices_isolate)]
    particle_pos[to_isolate,:] = np.ones(2)*-1 # Remove the isolated particle from the system


    # TEST AN AMOUNT OF CITIZENS, AND ISOLATE POSITIVES (this assumes that we don't always know who has the disease and thus cannot guarantee we are testing only the positive ones)
    if it%test_frequency==0:
        tested = np.zeros(N)
        tested[:int(test_percentage*N)]=1
        np.random.shuffle(tested)
        indices_to_isolate = np.intersect1d(infected_particles,np.where(tested==1)[0]) # check if this person was actually infected
        particle_pos[indices_to_isolate,:] = np.ones(2)*-1

    #UPDATE PLOT VARIABLES
    yS[it] = N_clean
    yI[it] = N_infected
    yR[it] = N_immune

    # plot particle positions
    if it%plotcounter==0:
        #plot particles
        scatter_plot.set_offsets(particle_pos)
        scatter_plot.set_facecolor(color_scheme[particle_infected])
        scatter_plot.set_edgecolor(color_scheme[particle_infected])
        #plot SIR graph
        ax2.cla()
        Iplot = ax2.fill_between(xSIR, 0, yI, color=color_scheme[1])
        Splot = ax2.fill_between(xSIR, yI, yI + yS, color=color_scheme[0])
        Rplot = ax2.fill_between(xSIR, yI + yS, yI + yS + yR, color=color_scheme[2])
        ax2.axis([0, it, 0, N+1])

        plt.draw()
        plt.pause(0.001)


    #print time-level
    print(it*dt)
