import pandapower as pp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the IEEE 33-bus data
df = pd.read_csv('ieee33bus.txt', sep=' ', header=None, names=['from', 'to', 'P', 'Q', 'R', 'X', 'C'])

def create_33bus_network():
    net = pp.create_empty_network()

    # Add buses
    for i in range(33):
        pp.create_bus(net, vn_kv=12.66, name=f"Bus {i+1}")

    # Add ext_grid
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

def add_upfc_like_energy_router(net, from_bus, to_bus):
    # Add a transformer with phase shifting capability between the buses
    pp.create_transformer_from_parameters(net, hv_bus=from_bus, lv_bus=to_bus, 
                                          sn_mva=1, vn_hv_kv=12.66, vn_lv_kv=12.66,
                                          vsc_percent=10, vscr_percent=0.1, 
                                          pfe_kw=0, i0_percent=0, shift_degree=0,
                                          vkr_percent=0.1, vk_percent=10)
    
    # Add a shunt element (reactor or capacitor) at the target bus for voltage control
    pp.create_shunt(net, bus=to_bus, q_mvar=0, p_mw=0, vn_kv=12.66)
    
    return to_bus

def adjust_energy_router(net, router_bus, target_voltage=1.0):
    # Get transformer and shunt indices
    trafo_idx = net.trafo.index[net.trafo.lv_bus == router_bus][0]
    shunt_idx = net.shunt.index[net.shunt.bus == router_bus][0]
    
    # Adjust shunt (reactive power) for voltage control
    v_pu = net.res_bus.loc[router_bus, 'vm_pu']
    q_mvar_adjust = 2 * (target_voltage - v_pu)  # Simple proportional control
    net.shunt.loc[shunt_idx, 'q_mvar'] += q_mvar_adjust
    
    # Adjust transformer phase shift for power flow control (if needed)
    p_mw = net.res_bus.loc[router_bus, 'p_mw']
    if abs(p_mw) > 0.01:  # If there's significant power flow
        shift_adjust = np.sign(p_mw)  # Increase or decrease shift based on power flow direction
        new_shift = net.trafo.loc[trafo_idx, 'shift_degree'] + shift_adjust
        net.trafo.loc[trafo_idx, 'shift_degree'] = np.clip(new_shift, -10, 10)

def run_pf_and_plot_side_by_side(net_before, net_after, title_before, title_after):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot voltage profile before adding the energy router
    bus_voltages_before = net_before.res_bus.vm_pu
    ax[0].plot(range(1, len(bus_voltages_before) + 1), bus_voltages_before, 'bo-')
    ax[0].set_title(title_before)
    ax[0].set_xlabel('Bus Number')
    ax[0].set_ylabel('Voltage (p.u.)')
    ax[0].grid(True)
    ax[0].set_ylim(0.9, 1.05)
    
    # Plot voltage profile after adding the energy router
    bus_voltages_after = net_after.res_bus.vm_pu
    ax[1].plot(range(1, len(bus_voltages_after) + 1), bus_voltages_after, 'ro-')
    ax[1].set_title(title_after)
    ax[1].set_xlabel('Bus Number')
    ax[1].grid(True)
    ax[1].set_ylim(0.9, 1.05)
    
    plt.tight_layout()
    plt.show()
    
    return bus_voltages_before, bus_voltages_after

# Create the network
net_before = create_33bus_network()

# Run initial power flow
pp.runpp(net_before)

# Add UPFC-like energy router between bus 18 (index 17) and bus 19 (index 18)
net_after = net_before.deepcopy()  # Copy the network for comparison
router_bus = add_upfc_like_energy_router(net_after, 17, 18)

# Run power flow iterations to adjust UPFC-like energy router
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

# Plot side by side
bus_voltages_before, bus_voltages_after = run_pf_and_plot_side_by_side(net_before, net_after,
                                                                      "IEEE 33-bus Test Case - Initial",
                                                                      "IEEE 33-bus Test Case - With Energy Router Between Bus 18 and 19")

# Detailed analysis of the changes made by the energy router
print("\nVoltage Changes Due to Energy Router:")
voltage_changes = bus_voltages_after - bus_voltages_before
for i, delta_v in enumerate(voltage_changes):
    print(f"Bus {i+1}: Voltage change = {delta_v:.4f} p.u.")

print("\nLine Loading Changes Due to Energy Router:")
line_loading_before = net_before.res_line['loading_percent']
line_loading_after = net_after.res_line['loading_percent']
loading_changes = line_loading_after - line_loading_before
for i, delta_load in enumerate(loading_changes):
    print(f"Line {i+1}: Loading change = {delta_load:.2f}%")
