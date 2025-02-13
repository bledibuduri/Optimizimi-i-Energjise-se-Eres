import pandas as pd
import pulp
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Leximi i të dhënave ===
# Lexojmë datasetin për Kosovën
kosovo_data = pd.read_csv("kosovo_wind.csv")
# Lexojmë datasetin për Maqedoninë e Veriut
macedonia_data = pd.read_csv("macedonia_wind.csv")

# Konvertojmë kolonën 'time' në formatin datetime
kosovo_data['time'] = pd.to_datetime(kosovo_data['time'])
macedonia_data['time'] = pd.to_datetime(macedonia_data['time'])

# Filtrojmë të dhënat për periudhën 2014-2022
kosovo_data = kosovo_data[(kosovo_data['time'].dt.year >= 2014) & (kosovo_data['time'].dt.year <= 2022)]
macedonia_data = macedonia_data[(macedonia_data['time'].dt.year >= 2014) & (macedonia_data['time'].dt.year <= 2022)]

# === 2. Krijimi i modelit të optimizimit ===
problemi = pulp.LpProblem("Optimizimi_Energjise_Eres", pulp.LpMaximize)

# Lista e kohës (bazuar në dataset)
koha = kosovo_data['time'].tolist()

# Krijimi i variablave të vendimmarrjes
ruajtja_km = pulp.LpVariable.dicts("Ruajtja_KM", koha, lowBound=0, cat="Continuous")  # Energjia e ruajtur nga Kosova në Maqedoni
ruajtja_mk = pulp.LpVariable.dicts("Ruajtja_MK", koha, lowBound=0, cat="Continuous")  # Energjia e ruajtur nga Maqedonia në Kosovë

# Krijimi i variablës binare për kufizimin
z = pulp.LpVariable.dicts("z", koha, cat="Binary")
M = 1000  # Një numër i madh për të vendosur kufizimet e sakta

# === 3. Krijimi i funksionit objektiv ===
# Maksimizimi i përdorimit të erës
problemi += pulp.lpSum(ruajtja_km[t] + ruajtja_mk[t] for t in koha)

# === 4. Kufizimet ===
for t in koha:
    # Kufizimi që energjia e ruajtur në një drejtim të jetë 0 nëse ruajtja në drejtim tjetër është e aktivizuar
    problemi += ruajtja_km[t] <= z[t] * M  # Nëse z[t] = 0, ruajtja_km është 0
    problemi += ruajtja_mk[t] <= (1 - z[t]) * M  # Nëse z[t] = 1, ruajtja_mk është 0
    
    # Kufizimi që nuk mund të ruhet më shumë energji sesa prodhohet
    problemi += ruajtja_km[t] <= kosovo_data.loc[kosovo_data['time'] == t, 'XK'].values[0]
    problemi += ruajtja_mk[t] <= macedonia_data.loc[macedonia_data['time'] == t, 'MK'].values[0]

# === 5. Zgjidhja e problemit ===
problemi.solve()

# === 6. Shfaqja e rezultateve ===
print("Statusi i zgjidhjes:", pulp.LpStatus[problemi.status])

# Ruajmë rezultatet në një DataFrame
rezultatet = pd.DataFrame({
    "Koha": koha,
    "Ruajtja KM": [ruajtja_km[t].varValue for t in koha],
    "Ruajtja MK": [ruajtja_mk[t].varValue for t in koha]
})

# Ruajmë rezultatet në një skedar CSV
rezultatet.to_csv("rezultatet_optimizimi.csv", index=False)

print("Rezultatet u ruajtën me sukses në 'rezultatet_optimizimi.csv'")

# === Vizualizimi i rezultateve ===

# 1. Vizualizimi i energjisë së ruajtur në Kosovë dhe Maqedoni në subgrafikë të ndarë
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Grafiku për energjinë e ruajtur nga Kosova
ax1.plot(rezultatet['Koha'], rezultatet['Ruajtja KM'], label='Ruajtja KM (Kosovë -> Maqedoni)', color='blue')
ax1.set_title("Energjia e Ruajtur nga Kosova në Maqedoni (2014-2022)")
ax1.set_xlabel("Koha")
ax1.set_ylabel("Energjia e Ruajtur (MWh)")
ax1.legend()
ax1.grid(True)

# Grafiku për energjinë e ruajtur nga Maqedonia
ax2.plot(rezultatet['Koha'], rezultatet['Ruajtja MK'], label='Ruajtja MK (Maqedoni -> Kosovë)', color='green')
ax2.set_title("Energjia e Ruajtur nga Maqedonia në Kosovë (2014-2022)")
ax2.set_xlabel("Koha")
ax2.set_ylabel("Energjia e Ruajtur (MWh)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()  # Siguron që subgrafikët të mos mbivendosen
plt.show()


# 2. Grafiku për krahasimin e energjisë së prodhuar nga era në Kosovë dhe Maqedoni në subgrafikë të ndarë
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Grafiku për energjinë e prodhuar nga era në Kosovë
ax1.plot(kosovo_data['time'], kosovo_data['XK'], label='Energjia e Prodhur nga Era (Kosovë)', color='blue')
ax1.set_title("Energjia e Prodhur nga Era në Kosovë (2014-2022)")
ax1.set_xlabel("Koha")
ax1.set_ylabel("Energjia e Prodhur (MWh)")
ax1.legend()
ax1.grid(True)

# Grafiku për energjinë e prodhuar nga era në Maqedoni
ax2.plot(macedonia_data['time'], macedonia_data['MK'], label='Energjia e Prodhur nga Era (Maqedoni)', color='green')
ax2.set_title("Energjia e Prodhur nga Era në Maqedoni (2014-2022)")
ax2.set_xlabel("Koha")
ax2.set_ylabel("Energjia e Prodhur (MWh)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()  # Siguron që subgrafikët të mos mbivendosen
plt.show()


# 3. Shpërndarja e energjisë së ruajtur për secilin drejtim
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Shpërndarja për energjinë e ruajtur nga Kosova
sns.histplot(rezultatet['Ruajtja KM'], kde=True, color='blue', label='Ruajtja KM', bins=30, ax=ax1)
ax1.set_title("Shpërndarja e Energjisë së Ruajtur nga Kosova në Maqedoni")
ax1.set_xlabel("Energjia e Ruajtur (MWh)")
ax1.set_ylabel("Frekuenca")
ax1.legend()

# Shpërndarja për energjinë e ruajtur nga Maqedonia
sns.histplot(rezultatet['Ruajtja MK'], kde=True, color='green', label='Ruajtja MK', bins=30, ax=ax2)
ax2.set_title("Shpërndarja e Energjisë së Ruajtur nga Maqedonia në Kosovë")
ax2.set_xlabel("Energjia e Ruajtur (MWh)")
ax2.set_ylabel("Frekuenca")
ax2.legend()

plt.tight_layout()  # Siguron që subgrafikët të mos mbivendosen
plt.show()


#Vizualizimi i varshmërisë mes energjisë së prodhuar dhe të ruajtur, të ndara në subgrafikë

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Grafiku i energjisë së prodhuar nga era në Kosovë dhe Maqedoni
ax1.plot(kosovo_data['time'], kosovo_data['XK'], label='Energjia e Prodhur nga Era (Kosovë)', color='tab:blue', linewidth=2)
ax1.plot(macedonia_data['time'], macedonia_data['MK'], label='Energjia e Prodhur nga Era (Maqedoni)', color='tab:orange', linewidth=2)
ax1.set_title("Energji e Prodhur nga Era në Kosovë dhe Maqedoni", fontsize=14)
ax1.set_xlabel("Koha", fontsize=12)
ax1.set_ylabel("Energjia e Prodhur (MWh)", fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, which='both', linestyle='--', linewidth=0.7)

# Grafiku i energjisë së ruajtur nga Kosova dhe Maqedonia
ax2.plot(rezultatet['Koha'], rezultatet['Ruajtja KM'], label='Ruajtja KM (Kosovë -> Maqedoni)', color='tab:green', linestyle='-', linewidth=2)
ax2.plot(rezultatet['Koha'], rezultatet['Ruajtja MK'], label='Ruajtja MK (Maqedoni -> Kosovë)', color='tab:red', linestyle='--', linewidth=2)
ax2.set_title("Energji e Ruajtur nga Kosova dhe Maqedonia", fontsize=14)
ax2.set_xlabel("Koha", fontsize=12)
ax2.set_ylabel("Energjia e Ruajtur (MWh)", fontsize=12)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, which='both', linestyle='--', linewidth=0.7)

plt.tight_layout()  # Siguron që subgrafikët të mos mbivendosen
plt.show()

