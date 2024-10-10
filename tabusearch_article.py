import pandapower as pp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

# Step 1: Create the IEEE 33-bus network
def create_33bus_network():
    net = pp.create_empty_network()

    # Add buses
    for i in range(33):
        pp.create_bus(net, vn_kv=12.66, name=f"Bus {i+1}")

    # Add external grid
    pp.create_ext_grid(net, bus=0, vm_pu=1.0)

    # Load IEEE 33-bus data from file (assuming data is in 'ieee33bus.txt')
    df = pd.read_csv('ieee33bus.txt', sep=' ', header=None, names=['from', 'to', 'P', 'Q', 'R', 'X', 'C'])
    
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

# Step 2: Define the cost functions based on the article
def calculate_tic(router_size_mva, dg_size_mw):
    # Router cost as a function of size (MVA)
    F1_router = 0.0003
    F2_router = -0.185
    F3_router = 158
    router_cost = F1_router * router_size_mva**2 + F2_router * router_size_mva + F3_router

    # DG cost as a function of size (MW)
    alpha_dg = 500  # Cost per MW
    dg_cost = alpha_dg * dg_size_mw

    return router_cost + dg_cost

def calculate_tgc(power_losses, active_generation, reactive_generation):
    # Power losses cost (using example values for price per kWh)
    loss_cost_per_kwh = 0.1  # USD/kWh
    plc = max(power_losses, 0) * loss_cost_per_kwh * 8760  # Annualized

    # Active power generation cost
    alpha_pg = 50  # USD/MWh
    pgc = max(active_generation, 0) * alpha_pg * 8760  # Annualized

    # Reactive power generation cost
    alpha_qg = 10  # USD/Mvarh
    qgc = max(reactive_generation, 0) * alpha_qg * 8760  # Annualized

    return plc + pgc + qgc

# Step 3: Add devices according to each strategy
def add_dg_units(net):
    dg_size = 5  # Example size in MW
    pp.create_sgen(net, bus=30, p_mw=dg_size, name="DG Unit")
    return dg_size

def add_energy_router(net, from_bus, to_bus):
    router_size_mva = 10  # Example size in MVA
    pp.create_transformer_from_parameters(net, hv_bus=from_bus, lv_bus=to_bus, 
                                          sn_mva=router_size_mva, vn_hv_kv=12.66, vn_lv_kv=12.66,
                                          vsc_percent=10, vscr_percent=0.1, 
                                          pfe_kw=0, i0_percent=0, shift_degree=0,
                                          vkr_percent=0.1, vk_percent=10)
    return router_size_mva

def add_combined_strategy(net):
    dg_size = add_dg_units(net)
    router_size = add_energy_router(net, 18, 19)
    return dg_size, router_size

# Step 4: Optimize using Tabu search for each strategy
def generate_candidates(net, strategy="Combined"):
    candidates = []
    for i in range(1, 32):
        net_copy = copy.deepcopy(net)
        if strategy == "Router":
            add_energy_router(net_copy, i, i+1)
        elif strategy == "DG":
            pp.create_sgen(net_copy, bus=i, p_mw=5)  # Example DG
        elif strategy == "Combined":
            add_energy_router(net_copy, i, i+1)
            pp.create_sgen(net_copy, bus=i, p_mw=5)
        candidates.append(net_copy)
    return candidates

def evaluate_solution(net, dg_size=5, router_size_mva=10):
    pp.runpp(net)
    # Get system power losses, active and reactive generation
    power_losses = net.res_line['pl_mw'].sum()
    active_generation = net.res_bus['p_mw'].sum()
    reactive_generation = net.res_bus['q_mvar'].sum()

    tic = calculate_tic(router_size_mva, dg_size)
    tgc = calculate_tgc(power_losses, active_generation, reactive_generation)
    return tic, tgc

def update_pareto_front(pareto_front, candidates):
    for candidate in candidates:
        tic, tgc = evaluate_solution(candidate)
        if is_non_dominated(tic, tgc, pareto_front):
            pareto_front.append((tic, tgc))

def is_non_dominated(tic, tgc, pareto_front):
    for existing_tic, existing_tgc in pareto_front:
        if existing_tic <= tic and existing_tgc <= tgc:
            return False
    return True

def tabu_search(net, max_iter=10, strategy="Combined"):
    pareto_front = []
    tabu_list = []

    for iteration in range(max_iter):
        candidate_solutions = generate_candidates(net, strategy)
        update_pareto_front(pareto_front, candidate_solutions)

        if candidate_solutions:
            tabu_list.append(candidate_solutions[0])

        print(f"Iteration {iteration}: Pareto Front Length = {len(pareto_front)}")

    return pareto_front

# Step 5: Plot results
def plot_pareto_front(pareto_front, title):
    tic_values = [tic for tic, tgc in pareto_front]
    tgc_values = [tgc for tic, tgc in pareto_front]

    plt.figure(figsize=(10, 6))
    plt.scatter(tic_values, tgc_values, color='b', label="Pareto Front")
    plt.title(f"Pareto Front: {title}")
    plt.xlabel("Total Investment Cost (USD)")
    plt.ylabel("Total Generation Cost (USD/year)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_network_configuration(net, strategy_name):
    plt.figure(figsize=(12, 8))
    pp.plotting.simple_plot(net)
    plt.title(f"Optimized Network Configuration: {strategy_name}")
    plt.show()

# Main function to run all strategies and display outputs
def main():
    net = create_33bus_network()

    # Strategy 1: DG Units only
    pareto_front_dg = tabu_search(net, strategy="DG")
    plot_pareto_front(pareto_front_dg, "DG Units Only")
    plot_network_configuration(net, "DG Units Only")

    # Strategy 2: Energy Routers only
    pareto_front_router = tabu_search(net, strategy="Router")
    plot_pareto_front(pareto_front_router, "Energy Routers Only")
    plot_network_configuration(net, "Energy Routers Only")

    # Strategy 3: Combined Strategy
    pareto_front_combined = tabu_search(net, strategy="Combined")
    plot_pareto_front(pareto_front_combined, "Combined Strategy")
    plot_network_configuration(net, "Combined Strategy")

if __name__ == "__main__":
    main()
