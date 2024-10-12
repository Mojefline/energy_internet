import pandapower as pp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import random

# Read IEEE 33-bus data
df = pd.read_csv('ieee33bus.txt', sep=' ', header=None, names=['from', 'to', 'P', 'Q', 'R', 'X', 'C'])

# Create IEEE 33-bus network
def create_33bus_network():
    net = pp.create_empty_network()

    # Add buses
    for i in range(33):
        pp.create_bus(net, vn_kv=12.66, name=f"Bus {i+1}")

    # Add external grid
    pp.create_ext_grid(net, bus=0, vm_pu=1.0)

    # Add lines and loads
    for _, row in df.iterrows():
        from_bus = int(row['from']) - 1
        to_bus = int(row['to']) - 1
        pp.create_line_from_parameters(net, from_bus=from_bus, to_bus=to_bus, length_km=1, 
                                       r_ohm_per_km=row['R'], x_ohm_per_km=row['X'], 
                                       c_nf_per_km=0, max_i_ka=1)
        if row['P'] != 0 or row['Q'] != 0:
            pp.create_load(net, bus=to_bus, p_mw=row['P']/1000, q_mvar=row['Q']/1000)
    
    return net

# Add UPFC-like energy router between buses
def add_upfc_like_energy_router(net, from_bus, to_bus):
    pp.create_transformer_from_parameters(net, hv_bus=from_bus, lv_bus=to_bus, 
                                          sn_mva=1, vn_hv_kv=12.66, vn_lv_kv=12.66,
                                          vsc_percent=10, vscr_percent=0.1, 
                                          pfe_kw=0, i0_percent=0, shift_degree=0,
                                          vkr_percent=0.1, vk_percent=10)
    
    # Add static generator for voltage control
    pp.create_sgen(net, bus=to_bus, p_mw=0, q_mvar=0, name="Energy Router", controllable=True)
    return to_bus

# Adjust energy router for voltage control
def adjust_energy_router(net, router_bus, target_voltage=1.0):
    trafo_idx = net.trafo.index[net.trafo.lv_bus == router_bus][0]
    sgen_idx = net.sgen.index[net.sgen.bus == router_bus][0]
    
    v_pu = net.res_bus.loc[router_bus, 'vm_pu']
    q_mvar_adjust = 2 * (target_voltage - v_pu)
    net.sgen.loc[sgen_idx, 'q_mvar'] += q_mvar_adjust
    
    p_mw = net.res_bus.loc[router_bus, 'p_mw']
    if abs(p_mw) > 0.01:
        shift_adjust = np.sign(p_mw)
        new_shift = net.trafo.loc[trafo_idx, 'shift_degree'] + shift_adjust
        net.trafo.loc[trafo_idx, 'shift_degree'] = np.clip(new_shift, -10, 10)

# Step 2: Define the cost functions
def calculate_tic(router_size_mva, dg_size_mw):
    F1_router = 0.0003
    F2_router = -0.185
    F3_router = 158
    router_cost = F1_router * router_size_mva**2 + F2_router * router_size_mva + F3_router

    alpha_dg = 500
    dg_cost = alpha_dg * dg_size_mw

    return router_cost + dg_cost

def calculate_tgc(power_losses, active_generation, reactive_generation):
    loss_cost_per_kwh = 0.1
    plc = max(power_losses, 0) * loss_cost_per_kwh * 8760

    alpha_pg = 50
    pgc = max(active_generation, 0) * alpha_pg * 8760

    alpha_qg = 10
    qgc = max(reactive_generation, 0) * alpha_qg * 8760

    return plc + pgc + qgc

# Step 3: Add DG Units and Energy Routers for optimization
def add_dg_units(net, bus_id, dg_size=5):
    pp.create_sgen(net, bus=bus_id, p_mw=dg_size, name="DG Unit")
    return dg_size

# Step 4: Genetic Algorithm Optimization with Energy Routers
def genetic_algorithm(net, population_size=20, generations=50):
    mutation_prob = 0.1
    crossover_prob = 0.8

    population = []
    for _ in range(population_size):
        individual = {
            'dg_location': np.random.randint(1, 32),
            'dg_size': np.random.uniform(1, 10),
            'router_from_bus': np.random.randint(1, 32),
            'router_to_bus': np.random.randint(1, 32)
        }
        population.append(individual)

    def evaluate_fitness(individual):
        net_copy = copy.deepcopy(net)
        add_dg_units(net_copy, individual['dg_location'], individual['dg_size'])
        add_upfc_like_energy_router(net_copy, individual['router_from_bus'], individual['router_to_bus'])
        pp.runpp(net_copy)
        power_losses = net_copy.res_line['pl_mw'].sum()
        active_generation = net_copy.res_bus['p_mw'].sum()
        reactive_generation = net_copy.res_bus['q_mvar'].sum()

        tic = calculate_tic(1, individual['dg_size'])  # router_size_mva is fixed at 1 in the provided function
        tgc = calculate_tgc(power_losses, active_generation, reactive_generation)
        fitness = tic + tgc  # Minimize total cost
        return fitness

    def select_parents(population, fitness_values):
        return random.choices(population, weights=[1/f for f in fitness_values], k=2)

    def crossover(parent1, parent2):
        if np.random.rand() < crossover_prob:
            cross_point = np.random.randint(1, 4)
            child1, child2 = parent1.copy(), parent2.copy()
            if cross_point == 1:
                child1['dg_location'], child2['dg_location'] = parent2['dg_location'], parent1['dg_location']
            elif cross_point == 2:
                child1['dg_size'], child2['dg_size'] = parent2['dg_size'], parent1['dg_size']
            elif cross_point == 3:
                child1['router_from_bus'], child2['router_from_bus'] = parent2['router_from_bus'], parent1['router_from_bus']
                child1['router_to_bus'], child2['router_to_bus'] = parent2['router_to_bus'], parent1['router_to_bus']
            return child1, child2
        return parent1, parent2

    def mutate(individual):
        if np.random.rand() < mutation_prob:
            mutation_point = np.random.randint(0, 4)
            if mutation_point == 0:
                individual['dg_location'] = np.random.randint(1, 32)
            elif mutation_point == 1:
                individual['dg_size'] = np.random.uniform(1, 10)
            elif mutation_point == 2:
                individual['router_from_bus'] = np.random.randint(1, 32)
            else:
                individual['router_to_bus'] = np.random.randint(1, 32)
        return individual

    for generation in range(generations):
        fitness_values = [evaluate_fitness(ind) for ind in population]
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitness_values)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = new_population[:population_size]
        best_fitness = min(fitness_values)
        print(f"Generation {generation}, Best Fitness: {best_fitness}")

    best_individual = population[np.argmin(fitness_values)]
    return best_individual

# Step 5: Plot results
def plot_network_configuration(net, strategy_name):
    static_coords = {
        0:(0, 30),1:(10, 30),3:(20, 30),5:(30, 30),8:(40, 30),11:(50, 30),14:(60, 30),16:(70, 30),18:(80, 30),20:(90, 30),22:(100, 30),24:(110, 30),26:(120, 30),28:(130, 30),29:(140, 30),30:(150, 30),31:(160, 30),32:(170, 30),2:(10, 60),4:(20, 60),7:(30, 60),10:(40, 60),13:(60, 80),15:(70, 80),17:(80, 80),19:(90, 80),21:(100, 80),23:(110, 80),25:(120, 80),27:(130, 80),6:(20, 0),9:(30, 80),12:(40, 80)
    }

    for i in range(len(net.bus)):
        net.bus_geodata.loc[i, 'x'] = static_coords.get(i, (0, 0))[0]
        net.bus_geodata.loc[i, 'y'] = static_coords.get(i, (0, 0))[1]

    plt.figure(figsize=(12, 8))
    pp.plotting.simple_plot(net, show_plot=False)
    
    for i, row in net.bus_geodata.iterrows():
        plt.text(row['x'], row['y'], f"Bus {i+1}", fontsize=9, color='blue')
        if i in net.sgen.index:
            plt.scatter(row['x'], row['y'], color='green', label='DG' if i == 0 else "")
        if i in net.trafo.index:
            plt.scatter(row['x'], row['y'], color='red', label='Energy Router' if i == 0 else "")
    
    plt.legend(loc="upper right")
    plt.title(f"Optimized Network Configuration: {strategy_name}")
    plt.show()

# Main function to run optimization and display results
def main():
    net = create_33bus_network()

    best_solution = genetic_algorithm(net)
    print(f"Best Solution: {best_solution}")

    add_dg_units(net, best_solution['dg_location'], best_solution['dg_size'])
    add_upfc_like_energy_router(net, best_solution['router_from_bus'], best_solution['router_to_bus'])

    plot_network_configuration(net, "GA Optimized Network")

if __name__ == "__main__":
    main()
