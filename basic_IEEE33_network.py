import pandapower as pp
import matplotlib.pyplot as plt
import networkx as nx

import pandapower.networks as pn

net33 = pn.case33bw()

def create_ieee33_network():
    # Create an empty network
    net = pp.create_empty_network()
    # Add buses
    for i in range(1, 34):
        pp.create_bus(net, vn_kv=12.66, name=f"Bus {i}")
    # Add lines
    line_data = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
        (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
        (16, 17), (17, 18), (2, 19), (19, 20), (20, 21), (21, 22), (3, 23),
        (23, 24), (24, 25), (6, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (30, 31), (31, 32), (32, 33)
    ]
    for from_bus, to_bus in line_data:
        pp.create_line_from_parameters(net, from_bus=from_bus-1, to_bus=to_bus-1, length_km=1, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
    # Add loads
    load_data = [
        (2, 100), (3, 90), (4, 120), (5, 60), (6, 60), (7, 200), (8, 200),
        (9, 60), (10, 60), (11, 45), (12, 60), (13, 60), (14, 120), (15, 60),
        (16, 60), (17, 60), (18, 90), (19, 90), (20, 90), (21, 90), (22, 90),
        (23, 90), (24, 420), (25, 420), (26, 60), (27, 60), (28, 60), (29, 120),
        (30, 200), (31, 150), (32, 210), (33, 60)
    ]
    for bus, p_kw in load_data:
        pp.create_load(net, bus=bus-1, p_mw=p_kw/1000, q_mvar=0)
    # Add external grid connection (slack bus)
    pp.create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0)
    return net

def plot_network(net):
    G = nx.Graph()
    
    # Add nodes
    for i, bus in net.bus.iterrows():
        G.add_node(i)
    
    # Add edges
    for _, line in net.line.iterrows():
        G.add_edge(line.from_bus, line.to_bus)
    
    # Set up the plot
    plt.figure(figsize=(20, 10))
    
    # Define fixed positions for nodes
    pos = {
        0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (4, 0), 5: (5, 0),
        6: (6, 0), 7: (7, 0), 8: (8, 0), 9: (9, 0), 10: (10, 0), 11: (11, 0),
        12: (12, 0), 13: (13, 0), 14: (14, 0), 15: (15, 0), 16: (16, 0), 17: (17, 0),
        18: (2, 1), 19: (3, 1), 20: (4, 1), 21: (5, 1),
        22: (3, -1), 23: (4, -1), 24: (5, -1),
        25: (6, 1), 26: (7, 1), 27: (8, 1), 28: (9, 1), 29: (10, 1),
        30: (11, 1), 31: (12, 1), 32: (13, 1)
    }
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=8, font_weight='bold')
    
    # Add load information
    for i, load in net.load.iterrows():
        bus = load.bus
        load_mw = load.p_mw
        x, y = pos[bus]
        plt.text(x, y-0.1, f'{load_mw:.2f} MW', 
                 horizontalalignment='center', fontsize=6)
    plt.title("IEEE 33-Bus Test System", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Create the network
# net = create_ieee33_network()

# # Plot the network
# plot_network(net)

# # Print some basic network information
# print(net)

# Plot the network
plot_network(net33)

# Print some basic network information
print(net33)