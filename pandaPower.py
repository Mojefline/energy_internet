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

def add_energy_router(net, bus):
    # Create a virtual bus for the energy router
    virtual_bus = pp.create_bus(net, vn_kv=12.66, name=f"ER Virtual Bus {bus+1}")
    
    # Add a very low impedance line between the original bus and virtual bus
    pp.create_line_from_parameters(net, from_bus=bus, to_bus=virtual_bus, length_km=0.001,
                                   r_ohm_per_km=0.01, x_ohm_per_km=0.01, c_nf_per_km=0, max_i_ka=10)
    
    # Add a static generator for voltage control
    pp.create_sgen(net, bus=virtual_bus, p_mw=0, q_mvar=0, name="ER Voltage Control", controllable=True)
    
    # Add an ideal transformer (phase shifter) for power flow control
    pp.create_transformer_from_parameters(net, hv_bus=bus, lv_bus=virtual_bus,
                                          sn_mva=10, vn_hv_kv=12.66, vn_lv_kv=12.66,
                                          vkr_percent=0, vk_percent=0.1, pfe_kw=0, i0_percent=0,
                                          shift_degree=0, tap_side="hv", tap_neutral=0,
                                          tap_min=-10, tap_max=10, tap_step_degree=1, tap_pos=0)
    
    return virtual_bus

def adjust_energy_router(net, router_bus, target_voltage=1.0):
    sgen_idx = net.sgen.index[net.sgen.bus == router_bus][0]
    trafo_idx = net.trafo.index[net.trafo.lv_bus == router_bus][0]
    
    # Adjust static generator for voltage control
    v_pu = net.res_bus.loc[router_bus, 'vm_pu']
    q_mvar_adjust = 2 * (target_voltage - v_pu)  # Simple proportional control
    net.sgen.loc[sgen_idx, 'q_mvar'] += q_mvar_adjust
    
    # Adjust transformer phase shift for power flow control (if needed)
    p_mw = net.res_bus.loc[router_bus, 'p_mw']
    if abs(p_mw) > 0.01:  # If there's significant power flow
        shift_adjust = np.sign(p_mw)  # Increase or decrease shift based on power flow direction
        new_shift = net.trafo.loc[trafo_idx, 'shift_degree'] + shift_adjust
        net.trafo.loc[trafo_idx, 'shift_degree'] = np.clip(new_shift, -10, 10)

def run_pf_and_plot(net, title):
    pp.runpp(net)
    
    # Get bus voltages
    bus_voltages = net.res_bus.vm_pu
    
    # Plot voltage profile
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(bus_voltages) + 1), bus_voltages, 'bo-')
    plt.title(title)
    plt.xlabel('Bus Number')
    plt.ylabel('Voltage (p.u.)')
    plt.grid(True)
    plt.ylim(0.9, 1.05)
    plt.show()
    
    # Print power flow results
    print(f"\n{title} Results:")
    print(net.res_bus[['vm_pu', 'p_mw', 'q_mvar']])
    print("\nLine Loading:")
    print(net.res_line[['loading_percent']])

# Create the network
net = create_33bus_network()

# Run initial power flow and plot
run_pf_and_plot(net, "IEEE 33-bus Test Case - Initial")

# Add energy router to bus 18 (index 17)
router_bus = add_energy_router(net, 17)

# Run power flow iterations to adjust energy router
max_iterations = 20
for i in range(max_iterations):
    pp.runpp(net)
    adjust_energy_router(net, router_bus)
    
    if abs(1.0 - net.res_bus.loc[router_bus, 'vm_pu']) < 0.001:
        print(f"Converged after {i+1} iterations")
        break
else:
    print("Did not converge within maximum iterations")

# Run final power flow with energy router and plot
run_pf_and_plot(net, "IEEE 33-bus Test Case - With Energy Router at Bus 18")

# Print the final energy router parameters
sgen_idx = net.sgen.index[net.sgen.bus == router_bus][0]
trafo_idx = net.trafo.index[net.trafo.lv_bus == router_bus][0]
print(f"\nEnergy Router Final Parameters:")
print(f"Static Generator Q: {net.sgen.loc[sgen_idx, 'q_mvar']:.4f} MVAr")
print(f"Transformer Phase Shift: {net.trafo.loc[trafo_idx, 'shift_degree']:.2f} degrees")