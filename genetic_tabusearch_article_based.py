import pandapower as pp
import pandapower.networks as pn
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

min_dgs=0
max_dgs=10
min_routers=0
max_routers=10

# Create IEEE 33-bus network
def create_33bus_network():
    net = pn.case33bw()
    return net

# Add UPFC-like energy router
def add_energy_router(net, target_bus, target_voltage=1.0):
    x, y = net.bus_geodata.loc[target_bus, ['x', 'y']]
    offset_distance = 0.01
    new_x = x + offset_distance
    new_y = y + offset_distance
    new_bus = pp.create_bus(net, vn_kv=12.66, name="Energy Router Virtual Bus", geodata=[new_x, new_y])
    pp.create_transformer_from_parameters(net, hv_bus=target_bus, lv_bus=new_bus, 
                                          sn_mva=1, vn_hv_kv=12.66, vn_lv_kv=12.66,
                                          vsc_percent=10, vscr_percent=0.1, 
                                          pfe_kw=0, i0_percent=0, shift_degree=0,
                                          vkr_percent=0.1, vk_percent=10)
    pp.create_sgen(net, bus=new_bus, p_mw=0, q_mvar=0, name="Energy Router", controllable=True)
    adjust_energy_router(net, new_bus, target_voltage)
    return new_bus

# Adjust energy router voltage control
def adjust_energy_router(net, router_bus, target_voltage=1.0):
    pp.runpp(net)
    trafo_idx = net.trafo.index[net.trafo.lv_bus == router_bus][0]
    sgen_idx = net.sgen.index[net.sgen.bus == router_bus][0]
    tolerance = 1e-3
    max_iterations = 50
    iteration = 0
    while iteration < max_iterations:
        v_pu = net.res_bus.loc[router_bus, 'vm_pu']
        q_mvar_adjust = 2 * (target_voltage - v_pu)
        net.sgen.loc[sgen_idx, 'q_mvar'] += q_mvar_adjust
        pp.runpp(net)
        v_pu_updated = net.res_bus.loc[router_bus, 'vm_pu']
        if abs(v_pu_updated - target_voltage) <= tolerance:
            break
        p_mw = net.res_bus.loc[router_bus, 'p_mw']
        if abs(p_mw) > 0.01:
            shift_adjust = np.sign(p_mw)
            new_shift = net.trafo.loc[trafo_idx, 'shift_degree'] + shift_adjust
            net.trafo.loc[trafo_idx, 'shift_degree'] = np.clip(new_shift, -10, 10)
        iteration += 1

# Add DG Units
def add_dg_units(net, bus_ids, dg_sizes):
    for bus_id, dg_size in zip(bus_ids, dg_sizes):
        pp.create_sgen(net, bus=bus_id, p_mw=dg_size, name="DG Unit")
    return dg_sizes

# Cost functions (Total Investment Cost - TIC, Total Generation Cost - TGC)
def calculate_tic(router_sizes_mva, dg_sizes_mw):
    F1_router = 0.0003
    F2_router = -0.185
    F3_router = 158
    router_cost = sum([F1_router * size**2 + F2_router * size + F3_router for size in router_sizes_mva])

    alpha_dg = 500
    dg_cost = sum([alpha_dg * size for size in dg_sizes_mw])

    return router_cost + dg_cost

def calculate_tgc(power_losses, active_generation, reactive_generation):
    loss_cost_per_kwh = 0.1
    plc = max(power_losses, 0) * loss_cost_per_kwh * 8760

    alpha_pg = 50
    pgc = max(active_generation, 0) * alpha_pg * 8760

    alpha_qg = 10
    qgc = max(reactive_generation, 0) * alpha_qg * 8760

    return plc + pgc + qgc

# Genetic Algorithm Optimization
def genetic_algorithm(net, scenario, population_size=20, generations=5, target_voltage=1.0, num_dgs=2, num_routers=2):
    mutation_prob = 0.1
    crossover_prob = 0.8
    population = []

    # Scenario settings: multiple DGs and/or multiple routers
    for _ in range(population_size):
        num_dgs = np.random.randint(min_dgs, max_dgs+1)
        num_routers = np.random.randint(min_routers, max_routers+1)
        individual = {
            'dg_locations': [np.random.randint(1, 32) for _ in range(num_dgs)],
            'dg_sizes': [np.random.uniform(1, 10) for _ in range(num_dgs)],
            'router_target_buses': [np.random.randint(1, 32) for _ in range(num_routers)]
        }
        population.append(individual)

    def evaluate_fitness(individual):
        net_copy = copy.deepcopy(net)
        if scenario == 1 or scenario == 3:  # Add DGs
            add_dg_units(net_copy, individual['dg_locations'], individual['dg_sizes'])
        if scenario == 2 or scenario == 3:  # Add energy routers
            for router_bus in individual['router_target_buses']:
                add_energy_router(net_copy, router_bus, target_voltage)
        
        pp.runpp(net_copy)
        power_losses = net_copy.res_line['pl_mw'].sum()
        active_generation = net_copy.res_bus['p_mw'].sum()
        reactive_generation = net_copy.res_bus['q_mvar'].sum()

        # Cost calculations
        tic = calculate_tic([1] * num_routers, individual['dg_sizes'])  # Router size is fixed at 1 for now
        tgc = calculate_tgc(power_losses, active_generation, reactive_generation)
        fitness = tic + tgc

        return fitness, tic, tgc

    def select_parents(population, fitness_values):
        return random.choices(population, weights=[1/f[0] for f in fitness_values], k=2)

    def crossover(parent1, parent2):
        child1, child2 = parent1.copy(), parent2.copy()
        if np.random.rand() < crossover_prob:
            cross_point = np.random.randint(1, 3)
            if cross_point == 1:
                child1['dg_locations'], child2['dg_locations'] = parent2['dg_locations'], parent1['dg_locations']
                child1['dg_sizes'], child2['dg_sizes'] = parent2['dg_sizes'], parent1['dg_sizes']
            else:
                child1['router_target_buses'], child2['router_target_buses'] = parent2['router_target_buses'], parent1['router_target_buses']
        return child1, child2

    def mutate(individual):
        if np.random.rand() < mutation_prob:
            mutation_point = np.random.randint(0, 2)
            if mutation_point == 0:  # Mutate DG locations or sizes
                individual['dg_locations'] = [np.random.randint(1, 32) for _ in range(num_dgs)]
                individual['dg_sizes'] = [np.random.uniform(1, 10) for _ in range(num_dgs)]
            else:  # Mutate energy router locations
                individual['router_target_buses'] = [np.random.randint(1, 32) for _ in range(num_routers)]
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
        best_fitness = min(fitness_values, key=lambda x: x[0])
        print(f"Generation {generation}, Best Fitness: {best_fitness[0]} (TIC: {best_fitness[1]}, TGC: {best_fitness[2]})")

    best_individual = population[np.argmin([f[0] for f in fitness_values])]
    return best_individual, best_fitness

# Plot voltage profiles for comparison
def plot_voltage_profiles(base_profile, profiles, labels):
    plt.figure(figsize=(10,6))
    
    buses = np.arange(1, len(base_profile) + 1)

    # Plot base profile
    plt.plot(buses, base_profile, label="Base Case", color='blue', linestyle='--')

    # Loop through all profiles and truncate or pad them to match 33 buses
    for profile, label in zip(profiles, labels):
        # If the profile has more than 33 buses, truncate it
        if len(profile) > len(buses):
            profile = profile[:len(buses)]
        # If the profile has fewer buses, pad it with the last value to match
        elif len(profile) < len(buses):
            profile = np.pad(profile, (0, len(buses) - len(profile)), 'edge')
        
        plt.plot(buses, profile, label=label)
    
    plt.xlabel("Bus Number")
    plt.ylabel("Voltage (p.u.)")
    plt.title("Voltage Profile Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot network configuration with DGs and Routers
def plot_network_with_devices(net, dg_locations=None, router_locations=None, title="Network Configuration"):
    pp.plotting.simple_plot(net, show_plot=False)

    # Add bus names
    for bus_id in net.bus.index:
        x, y = net.bus_geodata.loc[bus_id, ['x', 'y']]
        plt.text(x, y, f'Bus {bus_id}', fontsize=8, ha='right')

    # Plot DG locations
    if dg_locations:
        for dg_bus in dg_locations:
            x, y = net.bus_geodata.loc[dg_bus, ['x', 'y']]
            plt.scatter(x, y, color='green', s=100, label="DG", edgecolor='black')

    # Plot Router locations
    if router_locations:
        for router_bus in router_locations:
            x, y = net.bus_geodata.loc[router_bus, ['x', 'y']]
            plt.scatter(x, y, color='red', s=100, label="Energy Router", edgecolor='black')

    plt.title(title)
    plt.legend()
    plt.show()

# Main function to run optimization and scenarios
def main():
    net_base = create_33bus_network()
    pp.runpp(net_base)
    base_voltage_profile = net_base.res_bus['vm_pu'].values

    # Scenario 1: Optimize DG placement and size
    best_dg, best_fitness_dg = genetic_algorithm(net_base, scenario=1, num_dgs=2)
    net_dg_only = copy.deepcopy(net_base)
    add_dg_units(net_dg_only, best_dg['dg_locations'], best_dg['dg_sizes'])
    pp.runpp(net_dg_only)
    voltage_profile_dg = net_dg_only.res_bus['vm_pu'].values

    # Scenario 2: Optimize energy router placement
    best_router, best_fitness_router = genetic_algorithm(net_base, scenario=2, num_routers=2)
    net_router_only = copy.deepcopy(net_base)
    for router_bus in best_router['router_target_buses']:
        add_energy_router(net_router_only, router_bus)
    pp.runpp(net_router_only)
    voltage_profile_router = net_router_only.res_bus['vm_pu'].values

    # Scenario 3: Optimize both DG and energy router placement
    best_combination, best_fitness_combination = genetic_algorithm(net_base, scenario=3, num_dgs=2, num_routers=2)
    net_combined = copy.deepcopy(net_base)
    add_dg_units(net_combined, best_combination['dg_locations'], best_combination['dg_sizes'])
    for router_bus in best_combination['router_target_buses']:
        add_energy_router(net_combined, router_bus)
    pp.runpp(net_combined)
    voltage_profile_combined = net_combined.res_bus['vm_pu'].values

    # Plot voltage profiles for comparison
    plot_voltage_profiles(base_voltage_profile, 
                          [voltage_profile_dg, voltage_profile_router, voltage_profile_combined], 
                          ["DG Only", "Router Only", "DG + Router"])

    # Plot network configuration with devices for each scenario
    plot_network_with_devices(net_base, title="Base Network")
    plot_network_with_devices(net_dg_only, dg_locations=best_dg['dg_locations'], title="DG Optimized Network")
    plot_network_with_devices(net_router_only, router_locations=best_router['router_target_buses'], title="Router Optimized Network")
    plot_network_with_devices(net_combined, dg_locations=best_combination['dg_locations'], 
                              router_locations=best_combination['router_target_buses'], 
                              title="DG + Router Optimized Network")

    # Print best cost results and device details for each scenario
    print(f"Scenario 1 (DG Only): Best Fitness = {best_fitness_dg[0]}, TIC = {best_fitness_dg[1]}, TGC = {best_fitness_dg[2]}")
    print(f"Best DG locations: {best_dg['dg_locations']} with sizes: {best_dg['dg_sizes']}")
    
    print(f"Scenario 2 (Router Only): Best Fitness = {best_fitness_router[0]}, TIC = {best_fitness_router[1]}, TGC = {best_fitness_router[2]}")
    print(f"Best Router locations: {best_router['router_target_buses']}")

    print(f"Scenario 3 (DG + Router): Best Fitness = {best_fitness_combination[0]}, TIC = {best_fitness_combination[1]}, TGC = {best_fitness_combination[2]}")
    print(f"Best DG locations: {best_combination['dg_locations']} with sizes: {best_combination['dg_sizes']}")
    print(f"Best Router locations: {best_combination['router_target_buses']}")

if __name__ == "__main__":
    main()

