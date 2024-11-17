import random
import logging
from app.crossover.crossover_method import CrossoverMethod
from app.real_value_chromosome import RealValuedChromosome

class AveragingCrossover(CrossoverMethod):

    def __init__(self, gene_range):
        self.logger = logging.getLogger(__name__)
        self.gene_range = gene_range

    def crossover(self, chromosomes, expected_new_population_size):
        self.logger.debug("Performing averaging crossover on the chromosomes")
        new_population = []
        while len(new_population) < expected_new_population_size:
            parent1, parent2 = random.sample(chromosomes, 2)

            child_genes = [(gene1 + gene2) / 2 for gene1, gene2 in zip(parent1.genes, parent2.genes)]

            if all(self.gene_range[0] <= gene <= self.gene_range[1] for gene in child_genes):
                new_population.append(RealValuedChromosome(len(child_genes), genes=child_genes))
        return new_population