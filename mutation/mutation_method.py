from abc import abstractmethod, ABC


class MutationMethod(ABC):

    @abstractmethod
    def mutate(self, chromosomes_to_mutate):
        """
        Mutates the given chromosomes.

        Parameters:
        chromosomes_to_mutate (list): List of chromosomes to mutate.

        Returns:
        list: List of mutated chromosomes.
        """
        pass
