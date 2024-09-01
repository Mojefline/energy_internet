import pypsa
import pandas as pd

# Define a custom component class
class CustomStorage(pypsa.components.Component):
    name = "CustomStorage"
    list_name = "custom_storages"
    attrs = [
        ["bus", "string"],
        ["p_nom", "float"],
        ["efficiency_charge", "float"],
        ["efficiency_discharge", "float"],
        ["standing_loss", "float"],
        ["max_hours", "float"],
    ]
    status_vars = ["p", "state_of_charge"]

# Create a new PyPSA network
network = pypsa.Network()

# Add the custom component to the network
network.add("Carrier", "battery")
network.add("Bus", "bus1")
network.add("Generator", "gen1", bus="bus1", p_nom=100, marginal_cost=50)
network.add("Load", "load1", bus="bus1", p_set=80)

# Register the custom component
pypsa.descriptors.Dict.registry.update({CustomStorage.list_name: CustomStorage})
network.components.update({CustomStorage.list_name: CustomStorage})

# Add an instance of the custom component
network.add("CustomStorage", "storage1", bus="bus1", p_nom=50, efficiency_charge=0.95, efficiency_discharge=0.95, standing_loss=0.01, max_hours=4)

# Set up the optimization problem
network.optimize(solver_name="glpk")

# Print results
print(network.custom_storages_t.p)
print(network.custom_storages_t.state_of_charge)
print(network.generators_t.p)
print(network.loads_t.p)

# Plot the results
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

network.custom_storages_t.p.plot(ax=ax1, label="Storage Power")
network.generators_t.p.plot(ax=ax1, label="Generator Power")
network.loads_t.p.plot(ax=ax1, label="Load")
ax1.set_ylabel("Power (MW)")
ax1.legend()

network.custom_storages_t.state_of_charge.plot(ax=ax2, label="State of Charge")
ax2.set_ylabel("Energy (MWh)")
ax2.legend()

plt.tight_layout()
plt.show()