import pandas as pd
import pypsa

class EnergyRouter:
    def __init__(self, network):
        self.network = network
        self.df = pd.DataFrame(columns=['bus0', 'bus1', 'p_nom', 'p_nom_extendable', 'p_nom_min', 'p_nom_max', 'efficiency', 'capital_cost'])
        self.pnl = {'p0': pd.DataFrame(index=network.snapshots), 'p1': pd.DataFrame(index=network.snapshots)}

    def add(self, name, **kwargs):
        self.df.loc[name] = kwargs
        for attr in ['p0', 'p1']:
            self.pnl[attr].loc[:, name] = 0.0
    def remove(self, name):
        self.df.drop(name, inplace=True)
        for attr in self.pnl:
            if name in self.pnl[attr].columns:
                self.pnl[attr].drop(columns=[name], inplace=True)

    def copy(self, network):
        new_er = EnergyRouter(network)
        new_er.df = self.df.copy()
        new_er.pnl = {k: v.copy() for k, v in self.pnl.items()}
        return new_er

class EnhancedNetwork(pypsa.Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy_routers = EnergyRouter(self)

    def copy(self, with_time=True, *args, **kwargs):
        n = super().copy(with_time=with_time, *args, **kwargs)
        n.energy_routers = self.energy_routers.copy(n)
        return n