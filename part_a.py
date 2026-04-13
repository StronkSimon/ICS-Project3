"""
Part (a) – Single run of the SIR model.

Standard settings:
  N=500, K=10, recovery_chance=1%, spread_chance=10%
  No resistance (gain_resistance_chance=0)
  Start with 1 infected individual.

Produces:
  - fig_a_sir_timeseries.png  - S/I/R curves over time
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from sir_model import SIRModel, run_until_threshold, SUSCEPTIBLE, INFECTED, RECOVERED

# ── Parameters ────────────────────────────────────────────────────────────────
N = 500
K = 10
SPREAD_CHANCE    = 0.10
RECOVERY_CHANCE  = 0.01
THRESHOLD        = 0.90
MAX_STEPS        = 5_000

print("Part (a): Running single simulation …")
model = SIRModel(
    N=N,
    avg_node_degree=K,
    virus_spread_chance=SPREAD_CHANCE,
    recovery_chance=RECOVERY_CHANCE,
    gain_resistance_chance=0.0,
    initial_infected=1,
)

steps_to_90 = None
for step in range(1, MAX_STEPS + 1):
    model.step()
    if model.fraction_infected_or_recovered() >= THRESHOLD and steps_to_90 is None:
        steps_to_90 = step
    if not model.running:
        break

df = model.datacollector.get_model_vars_dataframe()

print(f"  Simulation ended at step {len(df)-1}")
if steps_to_90:
    print(f"  Steps to infect >90% of community: {steps_to_90}")
else:
    print("  90% threshold NOT reached within simulation window.")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
t = df.index

ax.fill_between(t, df["Susceptible"]/N*100, alpha=0.25, color="#3b82f6", label="_nolegend_")
ax.fill_between(t, df["Infected"]   /N*100, alpha=0.25, color="#ef4444", label="_nolegend_")
ax.fill_between(t, df["Recovered"]  /N*100, alpha=0.25, color="#22c55e", label="_nolegend_")

ax.plot(t, df["Susceptible"]/N*100, lw=2.5, color="#3b82f6", label="Susceptible (S)")
ax.plot(t, df["Infected"]   /N*100, lw=2.5, color="#ef4444", label="Infected (I)")
ax.plot(t, df["Recovered"]  /N*100, lw=2.5, color="#22c55e", label="Recovered (R)")

if steps_to_90:
    ax.axvline(steps_to_90, color="black", linestyle="--", lw=1.5,
               label=f"90% infected at step {steps_to_90}")

ax.set_xlabel("Time step", fontsize=13)
ax.set_ylabel("% of population", fontsize=13)
ax.set_title(
    f"Part (a) – SIR Model: Single Run\n"
    f"N={N}, K={K}, spread={SPREAD_CHANCE*100:.0f}%, recovery={RECOVERY_CHANCE*100:.0f}%",
    fontsize=13,
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(fontsize=11)
ax.set_xlim(left=0)
ax.set_ylim(0, 102)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_a_sir_timeseries.png", dpi=150)
print("  Saved: fig_a_sir_timeseries.png")
plt.close(fig)
