import pandas as pd
import numpy as np
import pypsa
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':

    # Read line data from CSV file
    df = pd.read_csv('ieee33_linedata.csv')

    # Convert 'From' and 'To' columns to integers and ensure they are strings in the network
    # df['From'] = df['From'].astype(int)
    # df['To'] = df['To'].astype(int)

    # Print the first few rows of the dataframe to check the data
    print("First few rows of the CSV data:")
    print(df.head())

    # System parameters
    V_base = 12.66  # kV
    S_base = 100  # MVA

    # Create PyPSA network
    network = pypsa.Network()
    network.set_snapshots(["now"])

    # Get unique bus numbers
    bus_numbers = sorted(set(df['From'].unique()) | set(df['To'].unique()))

    # Add buses
    for i in bus_numbers:
        network.add("Bus", str(i), v_nom=V_base)

    # Print added buses
    print("\nAdded buses:")
    print(network.buses)

    # Add lines
    for _, row in df.iterrows():
        network.add("Line", f"Line {int(row['From'])}-{int(row['To'])}",
                    bus0=str(int(row['From'])),
                    bus1=str(int(row['To'])),
                    r=row['R'],
                    x=row['X'],
                    s_nom=S_base)

    # Add loads
    for _, row in df.iterrows():
        if row['P_kW'] != 0 or row['Q_kVAr'] != 0:
            network.add("Load", f"Load {int(row['To'])}",
                        bus=str(int(row['To'])),
                        p_set=row['P_kW'] / 1000,
                        q_set=row['Q_kVAr'] / 1000)

    # Set Bus 1 as slack
    network.add("Generator", "Slack", bus="1", control="Slack", p_nom=1000, q_set=0)  # Set q_set to 0 for slack

    # Print network summary
    print("\nNetwork summary:")
    print(network)
    network.export_to_csv_folder("IEEE33")

    def calculate_voltage_deviation(network):
        return np.sum(np.abs(np.abs(network.buses_t.v_mag_pu.iloc[0]) - 1.0))


    def add_routers(network, router_buses):
        for bus in router_buses:
            try:
                network.add("Generator", f"Router at Bus {bus}",
                            bus=str(bus),
                            control="PV",
                            p_nom=100,
                            v_nom=1.0)
            except:
                print(f"Router at Bus {bus} already exists")


    def genetic_algorithm(network, n_routers, population_size=5, generations=5):
        bus_list = [str(b) for b in bus_numbers if b != 1]  # Exclude Bus 1 (slack)

        def create_individual():
            return random.sample(bus_list, n_routers)

        def crossover(parent1, parent2):
            crossover_point = random.randint(1, n_routers - 1)
            child = parent1[:crossover_point] + [x for x in parent2 if x not in parent1[:crossover_point]]
            return child[:n_routers]

        def mutate(individual):
            i = random.randint(0, n_routers - 1)
            individual[i] = random.choice(bus_list)
            return individual

        population = [create_individual() for _ in range(population_size)]

        for _ in range(generations):
            print(f"++++++++++++++++++generation: {_}")
            fitness_scores = []
            for ind in population:
                temp_network = network.copy()
                add_routers(temp_network, ind)
                try:
                    temp_network.pf()  # Run power flow
                    fitness = 1 / calculate_voltage_deviation(temp_network)
                    fitness_scores.append(fitness)
                except Exception as e:
                    print(f"Error in power flow calculation: {e}")
                    fitness_scores.append(0)  # Assign worst fitness if power flow fails

            if sum(fitness_scores) == 0:
                print("All individuals failed power flow. Restarting optimization.")
                return genetic_algorithm(network, n_routers, population_size, generations)

            new_population = []
            while len(new_population) < population_size:
                # print(f"new_population length: {len(new_population)}, {population_size}")
                parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)
                child = crossover(parent1, parent2)
                if random.random() < 0.1:  # 10% mutation rate
                    child = mutate(child)
                new_population.append(child)

            population = new_population

        best_individual = max(population, key=lambda ind: 1 / calculate_voltage_deviation(network))
        return best_individual


    def plot_voltage_profile(V_before, V_after, router_locations):
        plt.figure(figsize=(12, 6))

        plt.plot(range(1, len(V_before) + 1), V_before, 'b-', label='Before router placement')
        plt.plot(range(1, len(V_after) + 1), V_after, 'r-', label='After router placement')

        # for bus in router_locations:
        #     plt.axvline(x=bus, color='g', linestyle='--', label='Router location' if bus == router_locations[0] else '')

        plt.xlabel('Bus number')
        plt.ylabel('Voltage magnitude (p.u.)')
        plt.title('Voltage Profile Before and After Router Placement')
        plt.legend()
        plt.grid(True)
        plt.ylim(0.9, 1.05)
        plt.tight_layout()
        plt.savefig('voltage_profile.png')
        plt.show()

    def plot_base_voltage_profile(V_before):
        plt.figure(figsize=(12, 6))

        plt.plot(range(1, len(V_before) + 1), V_before, 'b-', label='Before router placement')
        plt.xlabel('Bus number')
        plt.ylabel('Voltage magnitude (p.u.)')
        plt.title('base Voltage Profile')
        plt.legend()
        plt.grid(True)
        plt.ylim(0.9, 1.05)
        plt.tight_layout()
        plt.savefig('voltage_profile.png')
        plt.show()


    # Calculate base system voltages
    try:
        network.pf()  # Run power flow
        initial_V = network.buses_t.v_mag_pu.iloc[0].values
    except Exception as e:
        print(f"Error in initial power flow calculation: {e}")
        initial_V = np.ones(len(network.buses))  # Fallback to nominal voltage

    plot_base_voltage_profile(initial_V)

    # Optimize router placement
    n_routers = 3
    best_router_locations = genetic_algorithm(network, n_routers)

    # Add optimal routers and recalculate
    add_routers(network, best_router_locations)
    try:
        network.pf()  # Run power flow
        final_V = network.buses_t.v_mag_pu.iloc[0].values
    except Exception as e:
        print(f"Error in final power flow calculation: {e}")
        final_V = initial_V  # Fallback to initial voltages

    # Plot voltage profile
    plot_voltage_profile(initial_V, final_V, best_router_locations)

    # Print results
    print(f"Optimal router locations: {best_router_locations}")
    print("\nRouter capacities (in MVA):")
    for bus in best_router_locations:
        capacity = abs(network.generators_t.p.loc["now", f"Router at Bus {bus}"] +
                       1j * network.generators_t.q.loc["now", f"Router at Bus {bus}"])
        print(f"Bus {bus}: {capacity:.4f}")

    print("\nVoltage deviation: {:.4f}".format(calculate_voltage_deviation(network)))
