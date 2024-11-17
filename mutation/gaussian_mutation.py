import random
import numpy as np
from app.mutation.mutation_method import MutationMethod
from app.real_value_chromosome import RealValuedChromosome

class GaussianMutation(MutationMethod):
    def __init__(self, mutation_rate, gene_range, mean=0, stddev=1):
        self.mutation_rate = mutation_rate
        self.gene_range = gene_range
        self.mean = mean
        self.stddev = stddev

    def mutate(self, chromosomes_to_mutate):
        mutated_chromosomes = []
        for chromosome in chromosomes_to_mutate:
            new_genes = []
            for gene in chromosome.genes:
                if random.random() < self.mutation_rate:
                    new_gene = gene + np.random.normal(self.mean, self.stddev)
                    new_gene = max(min(new_gene, self.gene_range[1]), self.gene_range[0])  # Ensure gene is within range
                    new_genes.append(new_gene)
                else:
                    new_genes.append(gene)
            mutated_chromosomes.append(RealValuedChromosome(len(new_genes), genes=new_genes))
        return mutated_chromosomes