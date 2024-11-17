
import random
import logging
from app.crossover.crossover_method import CrossoverMethod
from app.real_value_chromosome import RealValuedChromosome

class AlphaBetaMixingCrossover(CrossoverMethod):

    def __init__(self, gene_range, alpha=0.5, beta=0.5):
        self.logger = logging.getLogger(__name__)
        self.gene_range = gene_range
        self.alpha = alpha
        self.beta = beta

    def crossover(self, chromosomes, expected_new_population_size):
        self.logger.debug(f"Performing alpha-beta mixing crossover on the chromosomes with alpha: {self.alpha} and beta: {self.beta}")
        new_population = []
        while len(new_population) < expected_new_population_size:
            parent1, parent2 = random.sample(chromosomes, 2)
            child1_genes = [self.alpha * gene1 + self.beta * gene2 + (1 - self.alpha - self.beta) * gene3 for gene1, gene2, gene3 in zip(parent1.genes, parent2.genes, parent1.genes)]
            child2_genes = [self.beta * gene1 + (1 - self.beta) * gene2 for gene1, gene2 in zip(parent1.genes, parent2.genes)]

            if all(self.gene_range[0] <= gene <= self.gene_range[1] for gene in child1_genes) and \
                    all(self.gene_range[0] <= gene <= self.gene_range[1] for gene in child2_genes):
                new_population.append(RealValuedChromosome(len(child1_genes), genes=child1_genes))
                if len(new_population) < expected_new_population_size:
                    new_population.append(RealValuedChromosome(len(child2_genes), genes=child2_genes))
        return new_population