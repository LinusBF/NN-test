from evolution.agent import Agent
from evolution.generation_manager import GenerationManager
from evolution.genotype import Genotype
from network.network import Network


class EvolutionManager:
    def __init__(self, topology):
        self.generations = 0
        self.topology = topology
        self.genotypes = []
        self.agents = []

    def start_evolution(self, nr_of_genotypes):
        self.genotypes = []
        temp_network = Network(self.topology)
        for i in range(nr_of_genotypes):
            new_genotype = Genotype([0] * temp_network.weight_count)
            new_genotype.set_rand_params(-5, 5)
            self.genotypes.append(new_genotype)
        self.start_evaluation()

    def start_evaluation(self):
        agents = []
        for g in self.genotypes:
            agents.append(Agent(g, self.topology))
        self.agents = agents

    def evaluate(self, inputs, correct):
        for a in self.agents:
            a.evaluate(inputs, correct)

    def evolve(self):
        new_genotypes = GenerationManager.get_better_population(self.genotypes)
        mutated_genotypes = EvolutionManager.mutate(new_genotypes)
        self.generations = self.generations + 1
        self.genotypes = mutated_genotypes

    def get_sorted_agents(self):
        return sorted(self.agents)

    @staticmethod
    def mutate(pop):
        mutated_pop = [pop[0], pop[1]]  # Don't mutate best two genotypes
        for genotype in pop[2:]:
            genotype.mutate()
            mutated_pop.append(genotype)
        return mutated_pop
