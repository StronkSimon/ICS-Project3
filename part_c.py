"""
Part (c) - Effect of social distancing.

Relevant parameters for social distancing:
  1. avg_node_degree (K)   - fewer contacts = fewer edges
  2. virus_spread_chance   - masks/distance reduce transmission probability
  Both together model comprehensive social distancing.

Experiments:
  C1 - Sweep K from 2 to 20, fixed spread=10%
  C2 - Sweep spread from 1% to 20%, fixed K=10
  C3 - Heatmap: K x spread_chance → avg steps to 90%

Produces:
  - fig_c1_vary_K.png
  - fig_c2_vary_spread.png
  - fig_c3_heatmap.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
from functools import partial

# Assuming sir_model.py is in the same directory
from sir_model import SIRModel, run_until_threshold

# --- Configuration ---
N         = 500
RUNS      = 10
THRESHOLD = 0.90
MAX_STEPS = 8_000

# --- 1. Worker Function ---
def run_single_simulation(spread, degree):
    """
    Executes a single SIR simulation run. 
    This function is what gets shipped to different CPU cores.
    """
    m = SIRModel(
        N=N,
        avg_node_degree=degree,
        virus_spread_chance=spread,
        recovery_chance=0.01,
        gain_resistance_chance=0.0,
        initial_infected=1,
    )
    t = run_until_threshold(m, threshold=THRESHOLD, max_steps=MAX_STEPS)
    return t if t is not None else MAX_STEPS

# --- 2. Parallel Orchestrator ---
def run_parallel_batch(spread, degree, executor, runs=RUNS):
    """
    Dispatches a batch of runs to the executor and calculates stats.
    """
    # Create a list of tasks for this specific parameter set
    futures = [executor.submit(run_single_simulation, spread, degree) for _ in range(runs)]
    
    # Collect results as they finish
    times = [f.result() for f in futures]
    
    mean_val = np.mean(times)
    std_val = np.std(times)
    reach_rate = sum(1 for t in times if t < MAX_STEPS) / runs
    return mean_val, std_val, reach_rate

# --- 3. Main Execution Block ---
def main():
    # Using 'with' ensures the process pool is cleaned up properly
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        # ── C1: vary K ────────────────────────────────────────────────────────
        K_values = [2, 5, 10, 15, 20]
        print(f"C1: Varying avg_node_degree K (Parallel using {executor._max_workers} workers) ...")
        c1_mean, c1_std, c1_reach = [], [], []
        
        for k in K_values:
            m, s, r = run_parallel_batch(0.10, k, executor)
            c1_mean.append(m)
            c1_std.append(s)
            c1_reach.append(r)
            print(f"   K={k:2d}  avg={m:6.0f}  reach_rate={r:.2f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.errorbar(K_values, c1_mean, yerr=c1_std, fmt="o-", color="#3b82f6",
                     capsize=4, lw=2, markersize=7, label="Mean steps ± SD")
        ax1.set_xlabel("Average node degree K (contacts)", fontsize=12)
        ax1.set_ylabel("Steps to infect >90% population", fontsize=12)
        ax1.set_title("Effect of social distancing: varying K\n(spread=10%, recovery=1%)", fontsize=12)
        ax1.grid(alpha=0.3)
        ax1.legend()

        ax2.bar(K_values, c1_reach, color="#3b82f6", alpha=0.7, width=0.7)
        ax2.set_xlabel("Average node degree K (contacts)", fontsize=12)
        ax2.set_ylabel("Fraction of runs reaching 90%", fontsize=12)
        ax2.set_title("Fraction of runs reaching 90%\n(spread=10%, recovery=1%)", fontsize=12)
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig("fig_c1_vary_K.png", dpi=150)
        print("   Saved: fig_c1_vary_K.png")
        plt.close(fig)

        # ── C2: vary spread ───────────────────────────────────────────────────
        spread_values = [0.01, 0.05, 0.10, 0.15, 0.20]
        print("\nC2: Varying virus_spread_chance (Parallel) ...")
        c2_mean, c2_std, c2_reach = [], [], []
        
        for sp in spread_values:
            m, s, r = run_parallel_batch(sp, 10, executor)
            c2_mean.append(m)
            c2_std.append(s)
            c2_reach.append(r)
            print(f"   spread={sp:.2f}  avg={m:6.0f}  reach_rate={r:.2f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.errorbar([s*100 for s in spread_values], c2_mean, yerr=c2_std,
                     fmt="o-", color="#ef4444", capsize=4, lw=2, markersize=7)
        ax1.set_xlabel("Virus spread chance (%)", fontsize=12)
        ax1.set_ylabel("Steps to infect >90% population", fontsize=12)
        ax1.set_title("Effect of protective measures: varying spread chance\n(K=10, recovery=1%)", fontsize=12)
        ax1.grid(alpha=0.3)

        ax2.bar([s*100 for s in spread_values], c2_reach, color="#ef4444", alpha=0.7, width=0.8)
        ax2.set_xlabel("Virus spread chance (%)", fontsize=12)
        ax2.set_ylabel("Fraction of runs reaching 90%", fontsize=12)
        ax2.set_title("Fraction of runs reaching 90%\n(K=10, recovery=1%)", fontsize=12)
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig("fig_c2_vary_spread.png", dpi=150)
        print("   Saved: fig_c2_vary_spread.png")
        plt.close(fig)

        # ── C3: Heatmap K × spread ────────────────────────────────────────────
        K_grid   = [2, 5, 10, 15, 20]
        sp_grid  = [0.01, 0.05, 0.10, 0.15, 0.20]
        print(f"\nC3: Building {len(K_grid)}×{len(sp_grid)} heatmap (Parallel) ...")
        heat = np.zeros((len(sp_grid), len(K_grid)))
        heat_reach = np.zeros_like(heat)

        for i, sp in enumerate(sp_grid):
            for j, k in enumerate(K_grid):
                m, _, r = run_parallel_batch(sp, k, executor)
                heat[i, j] = m
                heat_reach[i, j] = r
                print(f"   spread={sp:.2f}, K={k} -> avg={m:.0f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        im1 = ax1.imshow(heat, aspect="auto", cmap="RdYlGn",
                         vmin=heat.min(), vmax=min(heat.max(), MAX_STEPS*0.8))
        ax1.set_xticks(range(len(K_grid)))
        ax1.set_xticklabels(K_grid)
        ax1.set_yticks(range(len(sp_grid)))
        ax1.set_yticklabels([f"{s*100:.0f}%" for s in sp_grid])
        ax1.set_xlabel("Average node degree K", fontsize=12)
        ax1.set_ylabel("Virus spread chance", fontsize=12)
        ax1.set_title("Mean steps to 90% infection\n(green=longer=better)", fontsize=12)
        for i in range(len(sp_grid)):
            for j in range(len(K_grid)):
                ax1.text(j, i, f"{heat[i,j]:.0f}", ha="center", va="center", fontsize=8,
                         color="black" if heat[i,j] < 6000 else "white")
        plt.colorbar(im1, ax=ax1, label="Steps")

        im2 = ax2.imshow(heat_reach, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
        ax2.set_xticks(range(len(K_grid)))
        ax2.set_xticklabels(K_grid)
        ax2.set_yticks(range(len(sp_grid)))
        ax2.set_yticklabels([f"{s*100:.0f}%" for s in sp_grid])
        ax2.set_xlabel("Average node degree K", fontsize=12)
        ax2.set_ylabel("Virus spread chance", fontsize=12)
        ax2.set_title("Fraction of runs reaching 90%\n(green=lower=better)", fontsize=12)
        for i in range(len(sp_grid)):
            for j in range(len(K_grid)):
                ax2.text(j, i, f"{heat_reach[i,j]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im2, ax=ax2, label="Fraction")

        fig.suptitle("Part (c) – Social Distancing: K × Spread Heatmap", fontsize=14)
        fig.tight_layout()
        fig.savefig("fig_c3_heatmap.png", dpi=150)
        print("   Saved: fig_c3_heatmap.png")
        plt.close(fig)

    print("\nPart (c) complete.")

if __name__ == "__main__":
    main()