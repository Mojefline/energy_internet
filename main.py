import numpy as np
import pandas as pd
import cmath

if __name__ == '__main__':

    # Read line data from CSV file
    df = pd.read_csv('ieee33_linedata.csv')

    # System parameters
    V_base = 12.66  # kV
    S_base = 100  # MVA
    Z_base = V_base ** 2 / S_base
    V_source = 1.0 + 0j  # p.u.

    # Initialize voltages
    num_buses = 33
    V = np.ones(num_buses, dtype=complex)
    V[0] = V_source

    # Create branch data structures
    branches = []
    S = np.zeros(num_buses, dtype=complex)

    for _, row in df.iterrows():
        from_bus, to_bus = int(row['From']), int(row['To'])
        R, X = row['R'] / Z_base, row['X'] / Z_base
        P, Q = row['P_kW'] / (1000 * S_base), row['Q_kVAr'] / (1000 * S_base)

        branches.append({
            'from': from_bus - 1,
            'to': to_bus - 1,
            'Z': complex(R, X)
        })

        S[to_bus - 1] = complex(P, Q)


    def backward_forward_sweep():
        max_iter = 100
        tolerance = 1e-6

        for _ in range(max_iter):
            V_prev = V.copy()

            # Backward sweep
            I = np.zeros(num_buses, dtype=complex)
            for branch in reversed(branches):
                to_bus = branch['to']
                I[to_bus] = np.conj(S[to_bus] / V[to_bus])
                I[branch['from']] += I[to_bus]

            # Forward sweep
            for branch in branches:
                from_bus, to_bus = branch['from'], branch['to']
                V[to_bus] = V[from_bus] - branch['Z'] * I[to_bus]

            if np.max(np.abs(V - V_prev)) < tolerance:
                break

        return V


    # Calculate bus voltages
    V = backward_forward_sweep()

    # Print results
    print("Bus Voltages:")
    for i, voltage in enumerate(V):
        mag, angle = cmath.polar(voltage)
        print(f"Bus {i + 1}: {mag:.4f} ∠ {np.degrees(angle):.2f}°")