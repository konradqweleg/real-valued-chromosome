from abc import ABC, abstractmethod

class FunctionToCalculate(ABC):
    @abstractmethod
    def calculate(self, variables):
        """
                Calculate the value of the function for the given variables
                :param variables: A list of variables
        """
        pass
