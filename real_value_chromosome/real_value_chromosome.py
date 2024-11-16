import random


class RealValuedChromosome:
    def __init__(self, num_parameters, genes=None, gene_range=None):
        """
        Konstruktor klasy reprezentującej chromosom rzeczywisty.
        :param genes: Lista wartości zmiennoprzecinkowych, które reprezentują geny chromosomu.
        :param num_parameters: Ilość parametrów funkcji która liczymy.
        :param gene_range: Zakres wartości dla genów (min, max), jeśli nie podano genes.
        """


        if genes:
            self.genes = genes
        elif gene_range:
            self.genes = [random.uniform(gene_range[0], gene_range[1]) for _ in range(num_parameters)]
        else:
            self.genes = [] * num_parameters

    def __str__(self):
        return f"Chromosome: {self.genes}"

