import random
import networkx as nx
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


# ── Agent states ──────────────────────────────────────────────────────────────
SUSCEPTIBLE = "S"
INFECTED    = "I"
RECOVERED   = "R"


class PersonAgent(Agent):
    def __init__(self, unique_id, model,
                 virus_spread_chance=0.10,
                 recovery_chance=0.01,
                 gain_resistance_chance=0.0,
                 check_frequency=1):
        super().__init__(unique_id, model)
        self.state = SUSCEPTIBLE
        self.virus_spread_chance    = virus_spread_chance
        self.recovery_chance        = recovery_chance
        self.gain_resistance_chance = gain_resistance_chance
        self.check_frequency        = check_frequency
        self.days_infected          = 0
        self.mutation_resistance    = False

    # ------------------------------------------------------------------
    def step(self):
        if self.state == INFECTED:
            self._try_infect_neighbors()
            self._try_recover()

    def _try_infect_neighbors(self):
        neighbors = list(self.model.grid.get_neighbors(self.pos, include_center=False))
        for neighbor in neighbors:
            if neighbor.state == SUSCEPTIBLE:
                spread = self.virus_spread_chance
                if hasattr(self.model, "mutation_active") and self.model.mutation_active:
                    spread = spread * self.model.mutation_spread_multiplier
                if self.random.random() < spread:
                    neighbor.state = INFECTED

    def _try_recover(self):
        if self.random.random() < self.recovery_chance:
            if self.random.random() < self.gain_resistance_chance:
                self.state = RECOVERED   # permanently resistant
            else:
                self.state = SUSCEPTIBLE  # back to susceptible


# ── Model ─────────────────────────────────────────────────────────────────────
class SIRModel(Model):
    """
    SIR epidemic model on a random network.

    Parameters
    ----------
    N                      : int   - number of nodes
    avg_node_degree        : int   - average contacts per person
    virus_spread_chance    : float - probability of infecting a neighbor per step
    recovery_chance        : float - probability of recovering per step
    gain_resistance_chance : float - probability of permanent immunity on recovery
    initial_infected       : int   - number of initially infected nodes
    mutation_active        : bool  - enable mutation mechanic (part e)
    mutation_spread_mult   : float - multiplier on spread chance for mutant strain
    mutation_resist_bypass : float - probability mutant bypasses built-up resistance
    """

    def __init__(self,
                 N=500,
                 avg_node_degree=10,
                 virus_spread_chance=0.10,
                 recovery_chance=0.01,
                 gain_resistance_chance=0.0,
                 initial_infected=1,
                 mutation_active=False,
                 mutation_spread_mult=1.0,
                 mutation_resist_bypass=0.0):

        super().__init__()
        self.N                       = N
        self.avg_node_degree         = avg_node_degree
        self.virus_spread_chance     = virus_spread_chance
        self.recovery_chance         = recovery_chance
        self.gain_resistance_chance  = gain_resistance_chance
        self.initial_infected        = initial_infected
        self.mutation_active         = mutation_active
        self.mutation_spread_multiplier = mutation_spread_mult
        self.mutation_resist_bypass  = mutation_resist_bypass

        self.schedule = RandomActivation(self)
        prob = avg_node_degree / N
        self.G = nx.erdos_renyi_graph(N, prob)

        # Mesa NetworkGrid
        from mesa.space import NetworkGrid
        self.grid = NetworkGrid(self.G)

        # Create agents
        for node in self.G.nodes():
            agent = PersonAgent(
                unique_id=node,
                model=self,
                virus_spread_chance=virus_spread_chance,
                recovery_chance=recovery_chance,
                gain_resistance_chance=gain_resistance_chance,
            )
            self.schedule.add(agent)
            self.grid.place_agent(agent, node)

        # Infect initial nodes
        infected_nodes = self.random.sample(list(self.G.nodes()), initial_infected)
        for node_id in infected_nodes:
            agents = self.grid.get_cell_list_contents([node_id])
            if agents:
                agents[0].state = INFECTED

        self.datacollector = DataCollector(
            model_reporters={
                "Susceptible": lambda m: self._count_state(SUSCEPTIBLE),
                "Infected":    lambda m: self._count_state(INFECTED),
                "Recovered":   lambda m: self._count_state(RECOVERED),
            }
        )
        self.datacollector.collect(self)
        self.running = True

    # ------------------------------------------------------------------
    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        if self._count_state(INFECTED) == 0:
            self.running = False

    def _count_state(self, state):
        return sum(1 for a in self.schedule.agents if a.state == state)

    def fraction_infected_or_recovered(self):
        n_i = self._count_state(INFECTED)
        n_r = self._count_state(RECOVERED)
        return (n_i + n_r) / self.N

    def fraction_infected(self):
        return self._count_state(INFECTED) / self.N


def run_until_threshold(model, threshold=0.90, max_steps=10_000):
    for step in range(1, max_steps + 1):
        model.step()
        if model.fraction_infected_or_recovered() >= threshold:
            return step
        if not model.running:
            break
    return None
