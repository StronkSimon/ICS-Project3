# ICS26 Project 3 – Spread of Infectious Disease
## SIR Agent-Based Model with Mesa

---

## Files

| File | Purpose |
|---|---|
| `sir_model.py` | Core model — `PersonAgent` + `SIRModel` (shared by all parts) |
| `part_a.py` | Single simulation run, S/I/R timeseries |
| `part_c.py` | Social distancing parameter sweep + heatmap |

---

## Requirements

```
pip install mesa==2.3.2 networkx matplotlib numpy pandas
```

## Output Figures

| Figure | Description |
|---|---|
| `fig_a_sir_timeseries.png` | S/I/R curves for a single standard run |
| `fig_c1_vary_K.png` | Effect of reducing social contacts K |
| `fig_c2_vary_spread.png` | Effect of reducing transmission probability |
| `fig_c3_heatmap.png` | Joint K × spread heatmap |

## Model Parameters

| Parameter | Default | Description |
|---|---|---|
| `N` | 500 | Number of nodes (people) |
| `avg_node_degree` (K) | 10 | Average contacts per person |
| `virus_spread_chance` | 0.10 | Per-contact transmission probability |
| `recovery_chance` | 0.01 | Probability of recovering each step |
| `gain_resistance_chance` | 0.0 | Prob. of permanent immunity on recovery |
| `initial_infected` | 1 | Number of infected at t=0 |

## Discussion Summaries

The large time differences between scenarios directly answer the two
questions from the assignment:
- **Preventive measures** (masks, reducing spread_chance) slow the epidemic
  significantly even without reducing contact numbers.
- **Lockdowns** (reducing K) further extend timelines because fewer network
  edges exist for the virus to traverse.

Together, these measures can delay the epidemic long enough for healthcare
systems to cope and for vaccines to be deployed.

Both K and spread_chance are relevant parameters for social distancing.
Below a critical value of K (the epidemic threshold), the virus fails to
sustain an outbreak in most runs. Combining low K with low spread_chance
has a compounding protective effect.
