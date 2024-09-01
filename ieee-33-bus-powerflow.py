import pandapower as pp
import pandas as pd
import numpy as np

def create_ieee33_network(csv_file):
    net = pp.create_empty_network()

    # Read line data from CSV
    line_data = pd.read_csv(csv_file)

    # Create buses
    for i in range(1, 34):  # 33 buses
        pp.create_bus(net, vn_kv=12.66, name=f"Bus {i}")

    # Create lines and loads
    for _, row in line_data.iterrows():
        from_bus = int(row['From']) - 1  # Adjusting for 0-based indexing
        to_bus = int(row['To']) - 1
        r = row['R']
        x = row['X']
        p_kw = row['P_kW'] / 1000  # Convert kW to MW
        q_kvar = row['Q_kVAr'] / 1000  # Convert kVAr to MVAr

        pp.create_line_from_parameters(net, from_bus=from_bus, to_bus=to_bus, length_km=1,
                                       r_ohm_per_km=r, x_ohm_per_km=x, c_nf_per_km=0, max_i_ka=1)
        
        # Add load to the 'to' bus
        pp.create_load(net, bus=to_bus, p_mw=p_kw, q_mvar=q_kvar)

    # Create external grid connection at bus 0
    pp.create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0)

    return net

def add_energy_router(net, from_bus, to_bus, p_mw, vm_pu=1.0, efficiency=0.98):
    new_bus = pp.create_bus(net, vn_kv=net.bus.vn_kv.at[from_bus], name=f"ER_Bus_{from_bus}_{to_bus}")
    pp.create_transformer(net, hv_bus=from_bus, lv_bus=new_bus, std_type="0.4 MVA 20/0.4 kV",
                          name=f"Energy Router {from_bus}-{to_bus}")
    net.bus.loc[new_bus, 'vm_pu'] = vm_pu
    pp.create_load(net, bus=new_bus, p_mw=p_mw, name=f"Energy Router Load {from_bus}-{to_bus}")
    pp.create_sgen(net, bus=from_bus, p_mw=p_mw/efficiency, q_mvar=0, name=f"Energy Router Gen {from_bus}-{to_bus}")
    pp.create_line(net, from_bus=new_bus, to_bus=to_bus, length_km=0.01, 
                   std_type="NAYY 4x50 SE", name=f"Energy Router Line {from_bus}-{to_bus}")

# Create the IEEE 33 bus system
net = create_ieee33_network('ieee33_linedata.csv')

# Add some energy routers (example)
add_energy_router(net, from_bus=5, to_bus=6, p_mw=0.5)
add_energy_router(net, from_bus=15, to_bus=16, p_mw=0.7)
add_energy_router(net, from_bus=25, to_bus=26, p_mw=0.6)

# Print network statistics
print("Network Statistics:")
print(f"Number of buses: {len(net.bus)}")
print(f"Number of lines: {len(net.line)}")
print(f"Number of loads: {len(net.load)}")
print(f"Number of transformers: {len(net.trafo)}")
print(f"Number of static generators: {len(net.sgen)}")
print(f"Total load: {net.load.p_mw.sum():.2f} MW, {net.load.q_mvar.sum():.2f} MVAr")

# Run power flow with adjusted parameters
try:
    pp.runpp(net, calculate_voltage_angles=True, enforce_q_lims=True, max_iteration=1000, 
             algorithm='nr', numba=True)
    print("\nPower flow converged successfully!")
except pp.powerflow.LoadflowNotConverged:
    print("\nPower flow did not converge. Diagnostic information:")
    print(f"Buses with voltage < 0.9 p.u.:")
    low_voltage_buses = net.res_bus[net.res_bus.vm_pu < 0.9]
    print(low_voltage_buses[['vm_pu']])
    print(f"\nBuses with voltage > 1.1 p.u.:")
    high_voltage_buses = net.res_bus[net.res_bus.vm_pu > 1.1]
    print(high_voltage_buses[['vm_pu']])
    print(f"\nLines with loading > 100%:")
    overloaded_lines = net.res_line[net.res_line.loading_percent > 100]
    print(overloaded_lines[['loading_percent']])

# Print results
print("\nBus Results:")
print(net.res_bus)
print("\nLine Results:")
print(net.res_line)
print("\nTransformer Results:")
print(net.res_trafo)
print("\nStatic Generator Results:")
print(net.res_sgen)
print("\nLoad Results:")
print(net.res_load)