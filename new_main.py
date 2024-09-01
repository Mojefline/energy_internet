import pandas as pd
import numpy as np
import pypsa
import matplotlib.pyplot as plt
import random
from energy_router_pypsaClass import EnhancedNetwork

def add_routers(network, router_buses):
    for i, bus in enumerate(router_buses):
        to_bus = f"ER_Bus_{i}"
        network.add("Bus", to_bus, v_nom=12.66)
        network.energy_routers.add(f"Router_{i}", 
                                   bus0=bus, bus1=to_bus, 
                                   p_nom=0, p_nom_extendable=True,
                                   p_nom_min=0, p_nom_max=10,  # MW
                                   capital_cost=100000,  # $/MW
                                   efficiency=0.95)

def calculate_voltage_deviation(network):
    return np.sum(np.abs(np.abs(network.buses_t.v_mag_pu.loc["now", :]) - 1.0))

def genetic_algorithm(network, n_routers, population_size=10, generations=10):
    bus_list = [str(b) for b in network.buses.index if b != "1"]  # Exclude Bus 1 (slack)

    def create_individual():
        return random.sample(bus_list, n_routers)

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, n_routers - 1)
        child = parent1[:crossover_point] + [x for x in parent2 if x not in parent1[:crossover_point]]
        return child[:n_routers]

    def mutate(individual):
        i = random.randint(0, n_routers - 1)
        individual[i] = random.choice([b for b in bus_list if b not in individual])
        return individual

    population = [create_individual() for _ in range(population_size)]

    for gen in range(generations):
        print(f"Generation: {gen + 1}")
        fitness_scores = []
        for ind in population:
            temp_network = network.copy()
            add_routers(temp_network, ind)
            try:
                temp_network.lopf(pyomo=False, solver_name='cbc')
                fitness = 1 / (calculate_voltage_deviation(temp_network) + 1e-6)
                fitness_scores.append(fitness)
            except Exception as e:
                print(f"Error in optimization: {e}")
                fitness_scores.append(0)  # Assign worst fitness if optimization fails

        if all(score == 0 for score in fitness_scores):
            print("All individuals failed optimization. Restarting generation.")
            population = [create_individual() for _ in range(population_size)]
            continue

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)
            child = crossover(parent1, parent2)
            if random.random() < 0.1:  # 10% mutation rate
                child = mutate(child)
            new_population.append(child)

        population = new_population

    best_individual = max(population, key=lambda ind: fitness_scores[population.index(ind)])
    return best_individual

def plot_voltage_profile(V_before, V_after, router_locations):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(V_before) + 1), V_before, 'b-', label='Before router placement')
    plt.plot(range(1, len(V_after) + 1), V_after, 'r-', label='After router placement')
    for bus in router_locations:
        plt.axvline(x=int(bus), color='g', linestyle='--', label='Router location' if bus == router_locations[0] else '')
    plt.xlabel('Bus number')
    plt.ylabel('Voltage magnitude (p.u.)')
    plt.title('Voltage Profile Before and After Router Placement')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.9, 1.05)
    plt.tight_layout()
    plt.savefig('voltage_profile.png')
    plt.show()

if __name__ == '__main__':
    # Load data and create network
    df = pd.read_csv('ieee33_linedata.csv')
    network = EnhancedNetwork()
    network.set_snapshots(["now"])

    # Add buses
    for i in range(1, 34):
        network.add("Bus", str(i), v_nom=12.66)

    # Add lines
    for _, row in df.iterrows():
        network.add("Line", f"Line {int(row['From'])}-{int(row['To'])}",
                    bus0=str(int(row['From'])),
                    bus1=str(int(row['To'])),
                    r=row['R'],
                    x=row['X'],
                    s_nom=100)

    # Add loads
    for _, row in df.iterrows():
        if row['P_kW'] != 0 or row['Q_kVAr'] != 0:
            network.add("Load", f"Load {int(row['To'])}",
                        bus=str(int(row['To'])),
                        p_set=row['P_kW'] / 1000,
                        q_set=row['Q_kVAr'] / 1000)

    # Add slack generator
    network.add("Generator", "Slack", bus="1", control="Slack", p_nom=1000)

    # Calculate initial voltage profile
    network.pf()
    initial_V = network.buses_t.v_mag_pu.loc["now", :].values

    # Optimize router placement
    n_routers = 3
    best_router_locations = genetic_algorithm(network, n_routers)

    # Add optimal routers and recalculate
    add_routers(network, best_router_locations)
    
    # Run optimization with routers
    network.optimize(solver_name='glpk')

    final_V = network.buses_t.v_mag_pu.loc["now", :].values

    # Plot voltage profile
    plot_voltage_profile(initial_V, final_V, best_router_locations)

    # Print results
    print(f"Optimal router locations: {best_router_locations}")
    print(f"Voltage deviation: {calculate_voltage_deviation(network):.4f}")

    # Print energy router power flows
    print("\nEnergy Router Power Flows:")
    for router in network.energy_routers.index:
        print(f"Bus {network.energy_routers.at[router, 'bus0']}: {network.energy_routers_t.p0.at['now', router]:.2f} MW")