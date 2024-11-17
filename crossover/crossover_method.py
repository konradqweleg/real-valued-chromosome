from abc import abstractmethod, ABC


class CrossoverMethod(ABC):
    @abstractmethod
    def crossover(self, chromosomes, expected_new_population_size, crossover_probability):
        """
                Perform crossover on the chromosomes
                :param chromosomes: A list of chromosomes
                :param expected_new_population_size: The expected size of the new population
        """
        pass
