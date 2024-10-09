import pandapower as pp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# Create IEEE 33-bus network
def create_33bus_network():
    net = pp.create_empty_network()

    # Add buses
    for i in range(33):
        pp.create_bus(net, vn_kv=12.66, name=f"Bus {i+1}")

    # Add external grid
    pp.create_ext_grid(net, bus=0, vm_pu=1.0)

    # Add lines and loads (simplified for now)
    df = pd.read_csv('ieee33bus.txt', sep=' ', header=None, names=['from', 'to', 'P', 'Q', 'R', 'X', 'C'])
    for _, row in df.iterrows():
        from_bus = int(row['from']) - 1
        to_bus = int(row['to']) - 1
        pp.create_line_from_parameters(net, from_bus=from_bus, to_bus=to_bus, length_km=1, 
                                       r_ohm_per_km=row['R'], x_ohm_per_km=row['X'], 
                                       c_nf_per_km=0, max_i_ka=1)
        if row['P'] != 0 or row['Q'] != 0:
            pp.create_load(net, bus=to_bus, p_mw=row['P']/1000, q_mvar=row['Q']/1000)
    
    return net

# Cost functions based on the article for energy router and DG
def calculate_tic(router_size_mva, dg_size_mw):
    # Router cost as a function of size (MVA)
    F1_router = 0.0003
    F2_router = -0.185
    F3_router = 158  # Base cost
    router_cost = F1_router * router_size_mva**2 + F2_router * router_size_mva + F3_router

    # DG cost as a function of size (MW)
    alpha_dg = 500  # Simplified per MW
    dg_cost = alpha_dg * dg_size_mw

    return router_cost + dg_cost

# Corrected TGC calculation
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

# Add DG units only (Strategy 1)
def add_dg_units(net):
    # Example function to add DG to bus and optimize its size
    dg_size = 5  # Example size
    pp.create_sgen(net, bus=30, p_mw=dg_size, name="DG Unit")
    return dg_size

# Add energy routers only (Strategy 2)
def add_energy_router(net, from_bus, to_bus):
    # Example UPFC-like energy router
    pp.create_transformer_from_parameters(net, hv_bus=from_bus, lv_bus=to_bus, 
                                          sn_mva=10, vn_hv_kv=12.66, vn_lv_kv=12.66,
                                          vsc_percent=10, vscr_percent=0.1, 
                                          pfe_kw=0, i0_percent=0, shift_degree=0,
                                          vkr_percent=0.1, vk_percent=10)
    return 10  # Example router size (MVA)

# Combined strategy with DGs and routers (Strategy 3)
def add_combined_strategy(net):
    dg_size = add_dg_units(net)
    router_size = add_energy_router(net, 18, 19)
    return dg_size, router_size

# Evaluate the solution: Run power flow and calculate costs
def evaluate_solution(net, dg_size=5, router_size_mva=10):
    pp.runpp(net)
    # Get system power losses, active and reactive generation
    power_losses = net.res_line['pl_mw'].sum()
    active_generation = net.res_bus['p_mw'].sum()
    reactive_generation = net.res_bus['q_mvar'].sum()

    tic = calculate_tic(router_size_mva, dg_size)
    tgc = calculate_tgc(power_losses, active_generation, reactive_generation)
    return tic, tgc

# Plot Pareto Front for TIC vs TGC
def plot_pareto_front(pareto_front):
    tic_values = []
    tgc_values = []
    
    for solution in pareto_front:
        tic, tgc = solution  # Simplified since solution is (TIC, TGC) tuple
        tic_values.append(tic)
        tgc_values.append(tgc)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(tic_values, tgc_values, color='b', label="Pareto Front")
    plt.title("Pareto Front: Total Investment Cost (TIC) vs Total Generation Cost (TGC)")
    plt.xlabel("Total Investment Cost (USD)")
    plt.ylabel("Total Generation Cost (USD/year)")
    plt.grid(True)
    plt.legend()
    plt.show()

# Tabu search algorithm for comparing strategies
def compare_strategies(net_before):
    pareto_front_dg = []
    pareto_front_router = []
    pareto_front_combined = []

    # Strategy 1: Only DG Units
    net_dg = net_before.deepcopy()
    dg_size = add_dg_units(net_dg)
    tic_dg, tgc_dg = evaluate_solution(net_dg, dg_size=dg_size, router_size_mva=0)
    pareto_front_dg.append((tic_dg, tgc_dg))

    # Strategy 2: Only Energy Routers
    net_router = net_before.deepcopy()
    router_size = add_energy_router(net_router, 18, 19)
    tic_router, tgc_router = evaluate_solution(net_router, dg_size=0, router_size_mva=router_size)
    pareto_front_router.append((tic_router, tgc_router))

    # Strategy 3: Combined DG and Routers
    net_combined = net_before.deepcopy()
    dg_size, router_size = add_combined_strategy(net_combined)
    tic_combined, tgc_combined = evaluate_solution(net_combined, dg_size=dg_size, router_size_mva=router_size)
    pareto_front_combined.append((tic_combined, tgc_combined))

    return pareto_front_dg, pareto_front_router, pareto_front_combined

# Create the network
net_before = create_33bus_network()

# Compare the three strategies
pareto_front_dg, pareto_front_router, pareto_front_combined = compare_strategies(net_before)

# Plot Pareto Fronts for each strategy
plot_pareto_front(pareto_front_dg)
plot_pareto_front(pareto_front_router)
plot_pareto_front(pareto_front_combined)

# Finalize and plot results for all strategies
def plot_all_strategies(pareto_front_dg, pareto_front_router, pareto_front_combined):
    plt.figure(figsize=(12, 8))
    
    # Unpack TIC and TGC values for each strategy
    tic_dg, tgc_dg = zip(*pareto_front_dg) if pareto_front_dg else ([], [])
    tic_router, tgc_router = zip(*pareto_front_router) if pareto_front_router else ([], [])
    tic_combined, tgc_combined = zip(*pareto_front_combined) if pareto_front_combined else ([], [])
    
    # Plot for DG Units
    plt.scatter(tic_dg, tgc_dg, color='blue', label='DG Units Only', alpha=0.5)

    # Plot for Energy Routers
    plt.scatter(tic_router, tgc_router, color='green', label='Energy Routers Only', alpha=0.5)

    # Plot for Combined Strategy
    plt.scatter(tic_combined, tgc_combined, color='red', label='Combined Strategy', alpha=0.5)

    # Add titles and labels
    plt.title("Pareto Front Comparison for Different Strategies")
    plt.xlabel("Total Investment Cost (TIC) - USD")
    plt.ylabel("Total Generation Cost (TGC) - USD/year")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Execute the function to plot all strategies
plot_all_strategies(pareto_front_dg, pareto_front_router, pareto_front_combined)

# Output detailed results for all strategies
def output_strategy_results(pareto_front, strategy_name):
    print(f"\nResults for {strategy_name}:")
    for tic, tgc in pareto_front:
        print(f"TIC: {tic:.2f}, TGC: {tgc:.2f}")

# Output results for each strategy
output_strategy_results(pareto_front_dg, "DG Units Only")
output_strategy_results(pareto_front_router, "Energy Routers Only")
output_strategy_results(pareto_front_combined, "Combined Strategy")
