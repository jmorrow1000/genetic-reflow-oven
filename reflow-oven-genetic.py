import numpy as np
import matplotlib.pyplot as plt
import pygad
import oven2_vectorized_np_dt1 as oven2

print(f'PyGad version: {pygad.__version__}')

#  belt_speed_factor = 0.00005  # adjusts belt speed gene value to belt_speed
max_speed = 0.012  #  max belt speed, m/sec
min_speed = 0.004  #  min belt speed, m/sec
max_htr = 400  # maximum allowed heater setting, deg-C
min_htr = 20  # minimum heater setting, deg-C
htr_element_lengths = np.array([0.225, 0.225, 0.225, 0.225, 0.225,
                                0.225, 0.225, 0.225, 1.750])  # meters
Thtr_init = np.zeros(len(htr_element_lengths) - 1)  # controllable heaters

#  Create product profile data points

num_profile_zones = 6
ambient_temp = 20  # degC
cool_temp = 20

def get_oven_pts(profile_pts, product_pts, ambient_temp):
    """ oven_pts is product_pts aligned to profile_pts
    """

    # array for oven readings set to length of profile
    oven_pts = np.full(len(profile_pts), ambient_temp)

    if len(product_pts) < len(profile_pts):
        oven_pts[0:len(product_pts)] = product_pts[0:len(product_pts)]

    else:
        oven_pts[0:len(profile_pts)] = product_pts[0:len(profile_pts)]

    return oven_pts

# Create product temperature-time profile
#   zone definition [start temp (degC), slope(degC/sec), duration (sec)]
#   start temp: profile temp. at beginning of a zone
#   slope: temp. change per sec over zone (=0 for constant temp.)
#   duration: length of zone in seconds (=None for zones using slope)
# profile steps are at 1 sec intervals

profile_zone_dT = [0 for i in range(num_profile_zones)]
product_profile = [0 for i in range(num_profile_zones)]

product_profile[0] = [ambient_temp, 2, None]  # zone 0
product_profile[1] = [150, 0, 60]  # zone 1
product_profile[2] = [150, 2, None]  # zone 2
product_profile[3] = [220, 0, 80]  # zone 3
product_profile[4] = [220, -2, None]  # zone 4
product_profile[5] = [100, 0, 2]  # zone 5

profile_list = [product_profile[0][0]]

for i in range(len(product_profile)):
    if product_profile[i][2] is None:  # zones defined by slope
        current_temp = product_profile[i][0]
        num_steps = abs((product_profile[i][0] - product_profile[i + 1][0]) / product_profile[i][1])
        profile_zone_dT[i] = int(num_steps)
        for j in range(int(num_steps)):
            current_temp += product_profile[i][1]
            profile_list.append(current_temp)

    else:  # zones defined by level
        profile_zone_dT[i] = int(product_profile[i][2])
        current_temp = product_profile[i][0]
        for j in range(product_profile[i][2]):
            profile_list.append(current_temp)

profile_pts = np.array(profile_list)
profile_size = len(profile_pts)

zone_ends = np.cumsum(profile_zone_dT)
run_time = zone_ends[-1]

#  plot profile
plt.plot(profile_pts[0:run_time])
plt.xlim(0,300)
plt.ylim(20, 250)
plt.xlabel('seconds')
plt.ylabel('deg-C')
plt.title("Product Profile Zones")
fnt_size = "small"
plt.text(40, 110,'zone 1', color='b', fontsize=fnt_size, horizontalalignment='right')
plt.text(85, 155,'zone 2', fontsize=fnt_size, color='b')
plt.text(140, 190,'zone 3', color='b', fontsize=fnt_size, horizontalalignment='right')
plt.text(190, 225,'zone 4', fontsize=fnt_size, color='b')
plt.text(260, 160,'zone 5', color='b', fontsize=fnt_size, horizontalalignment='right')
path = "/Users/johnrmorrow/downloads/genetic_plots/"
file_str = "desired_profile"+".png"
plt.savefig(path + file_str, dpi=300)
plt.show()

# initialize oven
oven = oven2.OvenClass(LH=htr_element_lengths)

#  PyGad setup

def fitness_func(ga_instance, solution, solution_idx):
    belt_speed = solution[8]
    oven_T = oven.run_oven(np.append(solution[0:8], cool_temp), belt_speed)
    oven_pts = get_oven_pts(profile_pts, oven_T, ambient_temp)
    fitness = -np.abs(profile_pts - oven_pts).sum() + 10000
    # '10000' is an arbitrary offset for plots w/ no effect on genetic calculations
    return fitness

fitness_function = fitness_func

def get_current_gen_num(ga_instance):
    print(f'generations completed: {ga_instance.generations_completed}')


num_generations = 200
num_parents_mating = 40

sol_per_pop = 200
num_genes = len(Thtr_init) + 1  # add a gene for belt speed

gene_space = [{'low': min_htr, 'high': max_htr}, {'low': min_htr, 'high': max_htr},
              {'low': min_htr, 'high': max_htr}, {'low': min_htr, 'high': max_htr},
              {'low': min_htr, 'high': max_htr}, {'low': min_htr, 'high': max_htr},
              {'low': min_htr, 'high': max_htr}, {'low': min_htr, 'high': max_htr},
              {'low': min_speed, 'high': max_speed}]

parent_selection_type = "sss"
keep_parents = 2  # active only if elitism = 0
keep_elitism = 1

crossover_type = "single_point"

mutation_type = "adaptive"
mutation_percent_genes = [40, 15]

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       save_best_solutions=True,
                       save_solutions=False,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       keep_elitism=keep_elitism,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=get_current_gen_num,
                       random_seed=4)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
best_solutions = ga_instance.best_solutions  # array: best solutions from all generations
print('***** best solution in final generation of run *****')
print(f'     heater settings: {np.array2string(solution[0:8], precision=0)}')
print(f'     belt_speed: {solution[8]:.4f}')
print(f'     fitness: {solution_fitness}')

best_generation = ga_instance.best_solution_generation  # generation with best solution
best_fitness = ga_instance.best_solutions_fitness  # fitness of best generation
print('***** best solution from best generation *****')
print(f'     best generation: {best_generation}')
best_htr = best_solutions[best_generation][0:8]
best_speed = best_solutions[best_generation][8]
print(f'     heater settings: {np.array2string(best_htr[0:8], precision=0)}')
print(f'     belt speed: {best_speed:.4f}')
print(f'     fitness: {best_fitness[best_generation]}')

def get_htr_boundaries(belt_speed):  # heater zone boundaries
    heater_deltas = htr_element_lengths[0:len(Thtr_init)] / belt_speed
    heater_ends = np.cumsum(heater_deltas)
    # idx_state = (heater_ends - heater_deltas / 2).astype(int)
    boundaries = np.append([0], heater_ends)
    return boundaries


#  sequence of generations plots
last_gen = best_solutions.shape[0]  # based on stop_criteria
#  last_gen = num_generations
for gen in range(0, last_gen, 1):
    Thtr = best_solutions[gen][0:8]
    belt_speed = best_solutions[gen][8]
    oven_T = oven.run_oven(np.append(Thtr, cool_temp), belt_speed)
    oven_pts = get_oven_pts(profile_pts, oven_T, ambient_temp)
    title_string = "Generation: " + str(gen)
    plt.plot(profile_pts[0:run_time], 'b', label='profile')
    plt.plot(oven_pts[0:run_time], 'r', label='genetic')
    plt.ylim(20, 250)
    plt.title(title_string)
    boundaries = get_htr_boundaries(belt_speed)
    plt.xticks(boundaries, ['   1→', '   2→', '   3→', '   4→',
                            '   5→', '   6→', '   7→', '   8→', ' '])
    plt.xlabel('heater zones')
    plt.ylabel('product temperature (deg-C)')
    plt.grid(axis='x', color='0.8')  # light gray
    plt.legend(loc="upper left")
    plt.text(20, 200, f'fitness: {best_fitness[gen]}')
    path = "/Users/johnrmorrow/downloads/genetic_plots/"
    file_str = "generation_" + str(gen) +".png"
    plt.savefig(path + file_str, dpi=300)
    plt.show()

#  best generation plot
Thtr = best_solutions[best_generation][0:8]
belt_speed = best_solutions[best_generation][8]
oven_T = oven.run_oven(np.append(Thtr, cool_temp), belt_speed)
oven_pts = get_oven_pts(profile_pts, oven_T, ambient_temp)
title_string = "Best Generation: " + str(ga_instance.best_solution_generation)
plt.plot(profile_pts[0:run_time], 'b', label='profile')
plt.plot(oven_pts[0:run_time], 'r', label='genetic')
plt.ylim(20, 250)
plt.title(title_string)
boundaries = get_htr_boundaries(belt_speed)
plt.xticks(boundaries, ['   1→', '   2→', '   3→', '   4→',
                        '   5→', '   6→', '   7→', '   8→', ' '])
plt.xlabel('heater zones')
plt.ylabel('product temperature (deg-C)')
plt.grid(axis='x', color='0.8')  # light gray
plt.legend(loc="upper left")
plt.text(20, 200, f'fitness: {best_fitness[best_generation]}')
path = "/Users/johnrmorrow/downloads/genetic_plots/"
file_str ="best_generation_" + str(best_generation) +".png"
plt.savefig(path + file_str, dpi=300)
plt.show()

#  fitness plot
plt.plot(best_fitness)
plt.plot(best_generation, best_fitness[best_generation],
         'ro', label="best generation")  # best fitness point
title_string = "Fitness"
plt.title(title_string)
plt.xlabel('generation')
plt.ylabel('fitness')
plt.legend(loc="lower right")
path = "/Users/johnrmorrow/downloads/genetic_plots/"
file_str ="fitness_plot" + ".png"
plt.savefig(path + file_str, dpi=300)
plt.show()
