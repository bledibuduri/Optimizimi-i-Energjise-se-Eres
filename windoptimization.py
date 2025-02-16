import pandas as pd
import pulp
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Reading Data ===
# Read the dataset for Kosovo
kosovo_data = pd.read_csv("kosovo_wind.csv")
# Read the dataset for North Macedonia
macedonia_data = pd.read_csv("macedonia_wind.csv")

# Convert the 'time' column to datetime format
kosovo_data['time'] = pd.to_datetime(kosovo_data['time'])
macedonia_data['time'] = pd.to_datetime(macedonia_data['time'])

# Filter data for the period 2014-2022
kosovo_data = kosovo_data[(kosovo_data['time'].dt.year >= 2014) & (kosovo_data['time'].dt.year <= 2022)]
macedonia_data = macedonia_data[(macedonia_data['time'].dt.year >= 2014) & (macedonia_data['time'].dt.year <= 2022)]

# === 2. Creating the Optimization Model ===
problem = pulp.LpProblem("Wind_Energy_Optimization", pulp.LpMaximize)

# Time list (based on dataset)
time_list = kosovo_data['time'].tolist()

# Creating decision variables
storage_km = pulp.LpVariable.dicts("Storage_KM", time_list, lowBound=0, cat="Continuous")  # Energy stored from Kosovo to Macedonia
storage_mk = pulp.LpVariable.dicts("Storage_MK", time_list, lowBound=0, cat="Continuous")  # Energy stored from Macedonia to Kosovo

# Creating binary variable for constraint enforcement
z = pulp.LpVariable.dicts("z", time_list, cat="Binary")
M = 1000  # Large number for accurate constraints

# === 3. Objective Function ===
# Maximizing wind energy utilization
problem += pulp.lpSum(storage_km[t] + storage_mk[t] for t in time_list)

# === 4. Constraints ===
for t in time_list:
    # Constraint: Stored energy in one direction should be 0 if the other direction is active
    problem += storage_km[t] <= z[t] * M  # If z[t] = 0, storage_km is 0
    problem += storage_mk[t] <= (1 - z[t]) * M  # If z[t] = 1, storage_mk is 0
    
    # Constraint: Cannot store more energy than produced
    problem += storage_km[t] <= kosovo_data.loc[kosovo_data['time'] == t, 'XK'].values[0]
    problem += storage_mk[t] <= macedonia_data.loc[macedonia_data['time'] == t, 'MK'].values[0]

# === 5. Solving the Problem ===
problem.solve()

# === 6. Displaying Results ===
print("Solution Status:", pulp.LpStatus[problem.status])

# Save results to a DataFrame
results = pd.DataFrame({
    "Time": time_list,
    "Storage KM": [storage_km[t].varValue for t in time_list],
    "Storage MK": [storage_mk[t].varValue for t in time_list]
})

# Save results to a CSV file
results.to_csv("optimization_results.csv", index=False)

print("Results successfully saved in 'optimization_results.csv'")

# === Visualization of Results ===

# 1. Visualization of stored energy from Kosovo and Macedonia in separate subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

ax1.plot(results['Time'], results['Storage KM'], label='Storage KM (Kosovo -> Macedonia)', color='blue')
ax1.set_title("Stored Energy from Kosovo to Macedonia (2014-2022)")
ax1.set_xlabel("Time")
ax1.set_ylabel("Stored Energy (MWh)")
ax1.legend()
ax1.grid(True)

ax2.plot(results['Time'], results['Storage MK'], label='Storage MK (Macedonia -> Kosovo)', color='green')
ax2.set_title("Stored Energy from Macedonia to Kosovo (2014-2022)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Stored Energy (MWh)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 2. Visualization of wind energy production in Kosovo and Macedonia
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

ax1.plot(kosovo_data['time'], kosovo_data['XK'], label='Wind Energy Production (Kosovo)', color='blue')
ax1.set_title("Wind Energy Production in Kosovo (2014-2022)")
ax1.set_xlabel("Time")
ax1.set_ylabel("Produced Energy (MWh)")
ax1.legend()
ax1.grid(True)

ax2.plot(macedonia_data['time'], macedonia_data['MK'], label='Wind Energy Production (Macedonia)', color='green')
ax2.set_title("Wind Energy Production in Macedonia (2014-2022)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Produced Energy (MWh)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 3. Distribution of stored energy for each direction
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

sns.histplot(results['Storage KM'], kde=True, color='blue', label='Storage KM', bins=30, ax=ax1)
ax1.set_title("Distribution of Stored Energy from Kosovo to Macedonia")
ax1.set_xlabel("Stored Energy (MWh)")
ax1.set_ylabel("Frequency")
ax1.legend()

sns.histplot(results['Storage MK'], kde=True, color='green', label='Storage MK', bins=30, ax=ax2)
ax2.set_title("Distribution of Stored Energy from Macedonia to Kosovo")
ax2.set_xlabel("Stored Energy (MWh)")
ax2.set_ylabel("Frequency")
ax2.legend()

plt.tight_layout()
plt.show()

# Heatmap of stored energy by month and year
results['Year'] = pd.to_datetime(results['Time']).dt.year
results['Month'] = pd.to_datetime(results['Time']).dt.month

pivot_km = results.pivot_table(index='Month', columns='Year', values='Storage KM', aggfunc='sum')
pivot_mk = results.pivot_table(index='Month', columns='Year', values='Storage MK', aggfunc='sum')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

sns.heatmap(pivot_km, cmap="Greens", annot=True, fmt=".0f", linewidths=0.5, ax=ax1)
ax1.set_title("Stored Energy from Kosovo (Heatmap)", fontsize=14)

sns.heatmap(pivot_mk, cmap="Reds", annot=True, fmt=".0f", linewidths=0.5, ax=ax2)
ax2.set_title("Stored Energy from Macedonia (Heatmap)", fontsize=14)

plt.tight_layout()
plt.show()




