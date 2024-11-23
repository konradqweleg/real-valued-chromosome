import random
from app.mutation.mutation_method import MutationMethod
from app.real_value_chromosome import RealValuedChromosome

class UniformMutation(MutationMethod):
    def __init__(self, mutation_rate, gene_range):
        self.mutation_rate = mutation_rate
        self.gene_range = gene_range


    def mutate(self, chromosomes_to_mutate):
        mutated_chromosomes = []
        for chromosome in chromosomes_to_mutate:
            new_genes = []
            for gene in chromosome.genes:
                if random.random() < self.mutation_rate:
                    new_gene = random.uniform(*self.gene_range)
                    new_gene = max(min(new_gene, self.gene_range[1]), self.gene_range[0])  # Ensure gene is within range
                    new_genes.append(new_gene)
                else:
                    new_genes.append(gene)
            mutated_chromosomes.append(RealValuedChromosome(len(new_genes), genes=new_genes))
        return mutated_chromosomes