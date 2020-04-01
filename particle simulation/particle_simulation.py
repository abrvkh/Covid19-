import numpy as np
import matplotlib.pyplot as plt

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
L_repel = 0.1 #this is the band in which particles are repelled from the boundary
N = 10 #population size
N_infected = 5 #number of initially infected people (should be smaller than N)
N_clean = N-N_infected #number of people that did not get the virus yet
N_immune = 0 #number of people that are immune
dt = 0.01 #time-step

##INFECTION PARAMETERS
infection_radius = 0.1 #people might get infected when they come within this distance of a infected particel
infection_probability = 0.01 #change of getting infected when within the radius
infection_time = 1 #time a person will be infected

#SOCIAL DISTANCEING
social_distance_radius = 0.1;

#INITIALISE PARTICLES
particle_pos = L*np.random.rand(N, 2) #positions
particle_vel = 2*(np.random.rand(N, 2)-0.5) #movement velocity
particle_infected = np.zeros(N, dtype=np.int) #indicates if a particle is infected or not (0=not infected, 1=infected, 2=immune)
particle_infectiontime = np.zeros(N) #indicates the time of when the particle was infected

#INFECT FIRST PARTICLES
for i in range(0, N_infected):
    index = np.random.randint(0, N) #pull a random integer (this represents the index of the infected particle
    particle_infected[index] = 1 #this particle is infected

#PLOT PARTICLES
plotcounter = 1 #plot every "plotcounter" frames
color_scheme = np.array(['blue', 'red', 'yellow']) #(clean, infected, immune) particle colours
scatter_plot = plt.scatter(particle_pos[:, 0], particle_pos[:, 1])
scatter_plot.set_sizes(10*np.ones(np.size(particle_pos, 1)))
scatter_plot.set_facecolor(color_scheme[particle_infected])
scatter_plot.set_edgecolor(color_scheme[particle_infected])
plt.axis([0, L, 0, L])

#TIME-LOOP
for it in range(1, 2000): #time loop

    #NEW POSITIONS
    for i in range(0, N): #loop over all particles to change position
        current_pos = particle_pos[i, :] #position of the particle
        current_vel = particle_vel[i, :] #velocity of the particle
        repulsion_force = np.zeros(2) #this is the repulsion force

        #new smooth velocity
        new_vel = randomwalk_step(current_pos, current_vel, dt)

        #repulsive force (social distancing)
        particle_dist = compute_distances(current_pos, particle_pos) #compute the distance between the current particle and the remaining particles
        particle_dist[particle_dist==0] = L #set the distance between itself to a large value
        close_particles_indices = np.where(particle_dist<social_distance_radius)[0]
        for j in close_particles_indices: #loop over all close particles to add a force
            vector = particle_pos[j] - current_pos #vector of the repulsive force
            norm = particle_dist[j] #this is the distance from the particle
            repulsion_force -= 0.001*vector/(norm**3)

        new_vel += repulsion_force

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

        particle_vel[i, :] = new_vel

    particle_pos += dt*particle_vel

    #NEW INFECTIONS
    infected_indices = np.where(particle_infected == 1)[0] #currently infected people
    for i in range(0, np.size(infected_indices)):
        infected_pos = particle_pos[infected_indices[i], :] #this is the current position of the infected particle
        in_danger_indices = np.where(np.linalg.norm(particle_pos-infected_pos, ord=2, axis=1) < infection_radius)[0] #find all the particles that are within the infection radius
        in_danger_indices = in_danger_indices[np.where(particle_infected[in_danger_indices]==0)[0]]#remove the indices of particles that are immune or are already infected
        new_infected_indices = in_danger_indices[np.random.rand(np.size(in_danger_indices)) < infection_probability] #these are the indices of particles that are infected
        particle_infected[new_infected_indices] = 1 #set these particles to infected
        particle_infectiontime[new_infected_indices] = it*dt #set the infection time to the current timelevel

    #NEW IMMUNE PEOPLE
    infected_particles = np.where(particle_infected==1)[0]
    new_immune_indices = infected_particles[np.where(it*dt - particle_infectiontime[infected_particles] > infection_time)[0]] #these are the infected people that are infected for a while
    particle_infected[new_immune_indices] = 2 #set these particles to immune

    # #plot particle positions
    if it%plotcounter==0:
        scatter_plot.set_offsets(particle_pos)
        scatter_plot.set_facecolor(color_scheme[particle_infected])
        scatter_plot.set_edgecolor(color_scheme[particle_infected])
        plt.show()
        plt.pause(0.001)


    #print time-level
    print(it*dt)