import pandapower as pp
import pandas as pd
from pandapower.auxiliary import _sum_by_group

def create_energy_router(net, bus, name=None, in_service=True, index=None):
    """
    Creates an energy router in the network.

    Parameters:
    -----------
    net : pandapower network
    bus : int
        The bus where the energy router is connected
    name : string, default None
        The name for this energy router
    in_service : boolean, default True
        Specifies if the energy router is in service.
    index : int, default None
        Force a specified ID if it is available

    Returns:
    --------
    index : int
        The unique ID of the created energy router
    """
    if index is None:
        index = pp.get_free_id(net["energy_router"])
        
    columns = ["name", "bus", "in_service"]
    variables = [name, bus, bool(in_service)]
    energy_router = pd.DataFrame(np.array(variables).reshape(1, -1), index=[index], columns=columns)
    
    if len(net["energy_router"]) == 0:
        net["energy_router"] = energy_router
    else:
        net["energy_router"] = pd.concat([net["energy_router"], energy_router])
    
    return index
def _pd_energy_router(net, energy_router_df):
    """
    Auxiliary function for power flow calculations.
    """
    ppc = net["_ppc"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]

    energy_router_buses = bus_lookup[energy_router_df["bus"].values]
    ppc["bus"][energy_router_buses, 7] = 1.0  # set voltage magnitude to 1.0 p.u.
    ppc["bus"][energy_router_buses, 1] = 1  # set bus type to PV bus

def _pf_energy_router(net, ppci, energy_router, bus, pq_buses):
    """
    Adjustment function for power flow calculations.
    """
    bus_idx = bus[energy_router["bus"]]
    ppci["bus"][bus_idx, 7] = 1.0
    ppci["bus"][bus_idx, 1] = 1
    pq_buses = np.setdiff1d(pq_buses, bus_idx)
    return ppci, pq_buses

# Add the energy router to the pandapower components
pp.create_std_type(element="energy_router", name="energy_router", data={"name": "energy_router"}, overwrite=True)
# Add the energy router to pandapower's power flow functions
pp.pd2ppc_callbacks.append(_pd_energy_router)
pp.runpp_3ph.pfsoln_callbacks.append(_pf_energy_router)