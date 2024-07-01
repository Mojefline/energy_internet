import numpy as np
import pandas as pd
import cmath
import random
import matplotlib.pyplot as plt

n_routers = 3  # desired number of routers

def backward_forward_sweep(router_buses):
    max_iter = 100
    tolerance = 1e-6

    for _ in range(max_iter):
        V_prev = V.copy()

        # Backward sweep
        I = np.zeros(num_buses, dtype=complex)
        for branch in reversed(branches):
            to_bus = branch['to']
            I[to_bus] = np.conj(S[to_bus] / V[to_bus])
            I[branch['from']] += I[to_bus]

        # Forward sweep
        for branch in branches:
            from_bus, to_bus = branch['from'], branch['to']
            if to_bus in router_buses:
                V[to_bus] = 1.0 + 0j  # Set voltage to 1 p.u. for router buses
            else:
                V[to_bus] = V[from_bus] - branch['Z'] * I[to_bus]

        if np.max(np.abs(V - V_prev)) < tolerance:
            break

    return V, I


def calculate_voltage_deviation(V):
    return np.sum(np.abs(np.abs(V) - 1.0))


def calculate_router_capacity(V, I, router_buses):
    capacities = {}
    for bus in router_buses:
        connected_branches = [b for b in branches if b['from'] == bus or b['to'] == bus]
        S_router = sum(V[b['to']] * np.conj(I[b['to']]) for b in connected_branches)
        capacities[bus] = abs(S_router)
    return capacities


def genetic_algorithm(n_routers, population_size=50, generations=100):
    def create_individual():
        return random.sample(range(1, num_buses), n_routers)

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, n_routers - 1)
        child = parent1[:crossover_point] + [x for x in parent2 if x not in parent1[:crossover_point]]
        return child[:n_routers]

    def mutate(individual):
        i = random.randint(0, n_routers - 1)
        individual[i] = random.randint(1, num_buses - 1)
        return individual

    population = [create_individual() for _ in range(population_size)]

    for _ in range(generations):
        fitness_scores = [1 / calculate_voltage_deviation(backward_forward_sweep(ind)[0]) for ind in population]

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)
            child = crossover(parent1, parent2)
            if random.random() < 0.1:  # 10% mutation rate
                child = mutate(child)
            new_population.append(child)

        population = new_population

    best_individual = max(population, key=lambda ind: 1 / calculate_voltage_deviation(backward_forward_sweep(ind)[0]))
    return best_individual


def plot_voltage_profile(V_before, V_after, router_locations):
    plt.figure(figsize=(12, 6))

    plt.plot(range(1, num_buses + 1), np.abs(V_before), 'b-', label='Before router placement')
    plt.plot(range(1, num_buses + 1), np.abs(V_after), 'r-', label='After router placement')

    for bus in router_locations:
        plt.axvline(x=bus + 1, color='g', linestyle='--',
                    label='Router location' if bus == router_locations[0] else '')

    plt.xlabel('Bus number')
    plt.ylabel('Voltage magnitude (p.u.)')
    plt.title('Voltage Profile Before and After Router Placement')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.9, 1.05)
    plt.tight_layout()
    plt.savefig('voltage_profile.png')
    plt.show()  # Add this line to display the plot
    plt.close()


def calculate_base_voltages():
    V_base = np.ones(num_buses, dtype=complex)
    V_base[0] = V_source

    for _ in range(100):  # Max iterations
        V_prev = V_base.copy()

        # Backward sweep
        I = np.zeros(num_buses, dtype=complex)
        for branch in reversed(branches):
            to_bus = branch['to']
            I[to_bus] = np.conj(S[to_bus] / V_base[to_bus])
            I[branch['from']] += I[to_bus]

        # Forward sweep
        for branch in branches:
            from_bus, to_bus = branch['from'], branch['to']
            V_base[to_bus] = V_base[from_bus] - branch['Z'] * I[to_bus]

        if np.max(np.abs(V_base - V_prev)) < 1e-6:
            break

    return V_base


if __name__ == '__main__':
    # Read line data from CSV file
    df = pd.read_csv('ieee33_linedata.csv')

    # System parameters
    V_base = 12.66  # kV
    S_base = 100  # MVA
    Z_base = V_base ** 2 / S_base
    V_source = 1.0 + 0j  # p.u.

    # Initialize voltages
    num_buses = 33
    V = np.ones(num_buses, dtype=complex)
    V[0] = V_source

    # Create branch data structures
    branches = []
    S = np.zeros(num_buses, dtype=complex)

    for _, row in df.iterrows():
        from_bus, to_bus = int(row['From']), int(row['To'])
        R, X = row['R'] / Z_base, row['X'] / Z_base
        P, Q = row['P_kW'] / (1000 * S_base), row['Q_kVAr'] / (1000 * S_base)

        branches.append({
            'from': from_bus - 1,
            'to': to_bus - 1,
            'Z': complex(R, X)
        })

        S[to_bus - 1] = complex(P, Q)

    # Calculate base system voltages (before any router placement)
    initial_V = calculate_base_voltages()

    # Optimize router placement
    best_router_locations = genetic_algorithm(n_routers)

    # Calculate final voltages and currents
    final_V, final_I = backward_forward_sweep(best_router_locations)

    # calculate routers capacities
    router_capacities = calculate_router_capacity(final_V, final_I, best_router_locations)

    # Plot voltage profile
    plot_voltage_profile(initial_V, final_V, best_router_locations)

    # Print voltage magnitudes for verification
    print("\nInitial Voltage Magnitudes:")
    print(np.abs(initial_V))
    print("\nFinal Voltage Magnitudes:")
    print(np.abs(final_V))

    # Print results
    print(f"Optimal router locations: {[bus + 1 for bus in best_router_locations]}")
    print("Router capacities (in p.u.):")
    for bus, capacity in router_capacities.items():
        print(f"Bus {bus + 1}: {capacity:.4f}")

    print("\nBus Voltages:")
    for i, voltage in enumerate(final_V):
        mag, angle = cmath.polar(voltage)
        print(f"Bus {i + 1}: {mag:.4f} ∠ {np.degrees(angle):.2f}°")

    print(f"\nVoltage deviation: {calculate_voltage_deviation(final_V):.4f}")

    print("\nVoltage profile plot has been saved as 'voltage_profile.png'")
