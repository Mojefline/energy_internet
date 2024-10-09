import pandapower as pp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

def calculate_tgc(power_losses, active_generation, reactive_generation):
    # Power losses cost (using example values for price per kWh)
    loss_cost_per_kwh = 0.1  # USD/kWh
    plc = power_losses * loss_cost_per_kwh * 8760  # Annualized

    # Active power generation cost
    alpha_pg = 50  # USD/MWh (example cost)
    pgc = active_generation * alpha_pg * 8760  # Annualized

    # Reactive power generation cost (assumed lower than active power)
    alpha_qg = 10  # USD/Mvarh
    qgc = reactive_generation * alpha_qg * 8760  # Annualized

    return plc + pgc + qgc

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

# Run power flow and plot side by side comparison
def run_pf_and_plot_side_by_side(net_before, net_after, title_before, title_after):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    bus_voltages_before = net_before.res_bus.vm_pu
    ax[0].plot(range(1, len(bus_voltages_before) + 1), bus_voltages_before, 'bo-')
    ax[0].set_title(title_before)
    ax[0].set_xlabel('Bus Number')
    ax[0].set_ylabel('Voltage (p.u.)')
    ax[0].grid(True)
    ax[0].set_ylim(0.9, 1.05)
    
    bus_voltages_after = net_after.res_bus.vm_pu
    ax[1].plot(range(1, len(bus_voltages_after) + 1), bus_voltages_after, 'ro-')
    ax[1].set_title(title_after)
    ax[1].set_xlabel('Bus Number')
    ax[1].grid(True)
    ax[1].set_ylim(0.9, 1.05)
    
    plt.tight_layout()
    plt.show()

# Generate candidate solutions by moving energy router to other bus locations
def generate_candidates(net):
    candidates = []
    for i in range(1, 32):
        net_copy = net.deepcopy()
        add_upfc_like_energy_router(net_copy, i, i+1)
        candidates.append(net_copy)
    return candidates

# Evaluate the solution: Run power flow and calculate costs
def evaluate_solution(net):
    pp.runpp(net)
    # Get system power losses, active and reactive generation
    power_losses = net.res_line['pl_mw'].sum()
    active_generation = net.res_bus['p_mw'].sum()
    reactive_generation = net.res_bus['q_mvar'].sum()

    # Assume router and DG sizes (example)
    router_size_mva = 10  # MVA
    dg_size_mw = 5  # MW
    
    tic = calculate_tic(router_size_mva, dg_size_mw)
    tgc = calculate_tgc(power_losses, active_generation, reactive_generation)
    return tic, tgc

# Update the Pareto front with non-dominated solutions
def update_pareto_front(pareto_front, candidates):
    for candidate in candidates:
        tic, tgc = evaluate_solution(candidate)
        if is_non_dominated(candidate, pareto_front):
            pareto_front.append(candidate)

# Check if a solution is non-dominated
def is_non_dominated(candidate, pareto_front):
    candidate_tic, candidate_tgc = evaluate_solution(candidate)
    for solution in pareto_front:
        tic, tgc = evaluate_solution(solution)
        if tic <= candidate_tic and tgc <= candidate_tgc:
            return False
    return True

# Tabu search algorithm
def tabu_search(net, max_iter=100):
    best_solution = None
    tabu_list = []
    pareto_front = []

    for iteration in range(max_iter):
        # Generate candidate solutions by placing the energy router in different locations
        candidate_solutions = generate_candidates(net)

        # Evaluate and update the Pareto front
        update_pareto_front(pareto_front, candidate_solutions)

        # Add the best solution to the Tabu list (first solution for simplicity)
        if candidate_solutions:
            best_solution = candidate_solutions[0]
            tabu_list.append(best_solution)

        print(f"Iteration {iteration}: Pareto Front Length = {len(pareto_front)}")

    return pareto_front

# Create the network
net_before = create_33bus_network()

# Run initial power flow
pp.runpp(net_before)

# Add UPFC-like energy router between bus 18 (index 17) and bus 19 (index 18)
net_after = net_before.deepcopy()  # Copy the network for comparison
router_bus = add_upfc_like_energy_router(net_after, 17, 18)

# Run power flow and adjust energy router
max_iterations = 20
for i in range(max_iterations):
    pp.runpp(net_after)
    adjust_energy_router(net_after, router_bus)
    
    if abs(1.0 - net_after.res_bus.loc[router_bus, 'vm_pu']) < 0.001:
        print(f"Converged after {i+1} iterations")
        break
else:
    print("Did not converge within maximum iterations")

# Run final power flow with UPFC-like energy router
pp.runpp(net_after)

# Plot voltage profile before and after the router is added
run_pf_and_plot_side_by_side(net_before, net_after, 
                             "IEEE 33-bus Test Case - Initial",
                             "IEEE 33-bus Test Case - With Energy Router")

# Run Tabu search to optimize energy router placement
pareto_front = tabu_search(net_after)

# Plot Pareto Front for TIC vs TGC
def plot_pareto_front(pareto_front):
    tic_values = []
    tgc_values = []
    
    for solution in pareto_front:
        tic, tgc = evaluate_solution(solution)
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

# Plot the final Pareto Front
plot_pareto_front(pareto_front)

# Detailed analysis of the changes made by the energy router
def analyze_results(net_before, net_after):
    voltage_changes = net_after.res_bus.vm_pu - net_before.res_bus.vm_pu
    line_loading_changes = net_after.res_line['loading_percent'] - net_before.res_line['loading_percent']

    print("\nVoltage Changes Due to Energy Router:")
    for i, delta_v in enumerate(voltage_changes):
        print(f"Bus {i+1}: Voltage change = {delta_v:.4f} p.u.")

    print("\nLine Loading Changes Due to Energy Router:")
    for i, delta_load in enumerate(line_loading_changes):
        print(f"Line {i+1}: Loading change = {delta_load:.2f}%")

# Perform detailed analysis of the results
analyze_results(net_before, net_after)

# Display the final Pareto Front values
print("\nFinal Pareto Front Solutions:")
for solution in pareto_front:
    tic, tgc = evaluate_solution(solution)
    print(f"TIC: {tic:.2f}, TGC: {tgc:.2f}")
