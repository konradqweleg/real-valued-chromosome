
import random
import logging
from app.crossover.crossover_method import CrossoverMethod
from app.real_value_chromosome import RealValuedChromosome

class LinearCrossover(CrossoverMethod):

    def __init__(self, gene_range):
        self.logger = logging.getLogger(__name__)
        self.gene_range = gene_range

    def crossover(self, chromosomes, expected_new_population_size, crossover_probability):
        self.logger.debug("Performing linear crossover on the chromosomes")
        new_population = []
        while len(new_population) < expected_new_population_size:
            if random.random() < crossover_probability:
                parent1, parent2 = random.sample(chromosomes, 2)
                alpha = random.random()
                beta = random.random()
                gamma = random.random()

                child1_genes = [alpha * gene1 + beta * gene2 + (1 - alpha - beta) * gene3 for gene1, gene2, gene3 in zip(parent1.genes, parent2.genes, parent1.genes)]
                child2_genes = [gamma * gene1 + (1 - gamma) * gene2 for gene1, gene2 in zip(parent1.genes, parent2.genes)]

                if all(self.gene_range[0] <= gene <= self.gene_range[1] for gene in child1_genes) and \
                        all(self.gene_range[0] <= gene <= self.gene_range[1] for gene in child2_genes):
                    new_population.append(RealValuedChromosome(len(child1_genes), genes=child1_genes))
                    if len(new_population) < expected_new_population_size:
                        new_population.append(RealValuedChromosome(len(child2_genes), genes=child2_genes))
        return new_population