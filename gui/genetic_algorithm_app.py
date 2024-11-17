import os
import csv
from datetime import datetime
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from app.function.cec_2014_f1 import Cec2014F1
from app.mutation.gaussian_mutation import GaussianMutation
from app.mutation.uniform_mutation import UniformMutation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from app.crossover.arithmetic_crossover import ArithmeticCrossover
from app.crossover.linear_crossover import LinearCrossover
from app.crossover.alpha_mixing_crossover import AlphaMixingCrossover
from app.crossover.alpha_beta_mixing_crossover import AlphaBetaMixingCrossover
from app.crossover.averaging_crossover import AveragingCrossover  # Import AveragingCrossover
from app.function.szwefel import Szwefel
from app.mutation.random_resetting import RandomResetting
from app.selection.best_selection import BestSelection
from app.selection.tournament_selection import TournamentSelection
from app.selection.roulette_wheel_selection import RouletteWheelSelection
from app.genetic_algorithm.genetic_algorithm import GeneticAlgorithm


class GeneticAlgorithmApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Genetic Algorithm Configuration")
        self.run_count = 0
        self.first_run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        config_frame = ttk.Frame(main_frame, padding="10")
        config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        plot_frame = ttk.Frame(main_frame, padding="10")
        plot_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(config_frame, text="Number of Parameters:").grid(row=0, column=0, padx=10, pady=5)
        self.num_parameters = tk.IntVar(value=2)
        ttk.Entry(config_frame, textvariable=self.num_parameters).grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(config_frame, text="Gene Range (min, max):").grid(row=1, column=0, padx=10, pady=5)
        self.gene_range_min = tk.DoubleVar(value=-500)
        self.gene_range_max = tk.DoubleVar(value=500)
        ttk.Entry(config_frame, textvariable=self.gene_range_min).grid(row=1, column=1, padx=10, pady=5)
        ttk.Entry(config_frame, textvariable=self.gene_range_max).grid(row=1, column=2, padx=10, pady=5)

        ttk.Label(config_frame, text="Population Size:").grid(row=2, column=0, padx=10, pady=5)
        self.population_size = tk.IntVar(value=10)
        ttk.Entry(config_frame, textvariable=self.population_size).grid(row=2, column=1, padx=10, pady=5)

        ttk.Label(config_frame, text="Epochs:").grid(row=3, column=0, padx=10, pady=5)
        self.epochs = tk.IntVar(value=10000)
        ttk.Entry(config_frame, textvariable=self.epochs).grid(row=3, column=1, padx=10, pady=5)

        ttk.Label(config_frame, text="Mutation Method:").grid(row=4, column=0, padx=10, pady=5)
        self.mutation_method = ttk.Combobox(config_frame,
                                            values=["RandomResetting", "UniformMutation", "GaussianMutation"])
        self.mutation_method.grid(row=4, column=1, padx=10, pady=5)
        self.mutation_method.current(0)

        ttk.Label(config_frame, text="Mutation Probability:").grid(row=4, column=2, padx=10, pady=5)
        self.mutation_probability = tk.DoubleVar(value=0.01)
        ttk.Entry(config_frame, textvariable=self.mutation_probability).grid(row=4, column=3, padx=10, pady=5)

        ttk.Label(config_frame, text="Selection Method:").grid(row=6, column=0, padx=10, pady=5)
        self.selection_method = ttk.Combobox(config_frame,
                                             values=["BestSelection", "TournamentSelection", "RouletteWheelSelection"])
        self.selection_method.grid(row=6, column=1, padx=10, pady=5)
        self.selection_method.current(0)

        ttk.Label(config_frame, text="Percentage of Best to Select:").grid(row=6, column=2, padx=10, pady=5)
        self.percentage_best_to_select = tk.DoubleVar(value=0.5)
        ttk.Entry(config_frame, textvariable=self.percentage_best_to_select).grid(row=6, column=3, padx=10, pady=5)

        ttk.Label(config_frame, text="Tournament Size (TournamentSelection):").grid(row=6, column=4, padx=10, pady=5)
        self.tournament_size = tk.IntVar(value=3)
        ttk.Entry(config_frame, textvariable=self.tournament_size).grid(row=6, column=5, padx=10, pady=5)



        ttk.Label(config_frame, text="Crossover Method:").grid(row=7, column=0, padx=10, pady=5)
        self.crossover_method = ttk.Combobox(config_frame,
                                             values=["ArithmeticCrossover", "LinearCrossover", "AlphaMixingCrossover",
                                                     "AlphaBetaMixingCrossover", "AveragingCrossover"])
        self.crossover_method.grid(row=7, column=1, padx=10, pady=5)
        self.crossover_method.current(0)

        ttk.Label(config_frame, text="Crossover Probability:").grid(row=7, column=2, padx=10, pady=5)
        self.crossover_probability = tk.DoubleVar(value=0.5)
        ttk.Entry(config_frame, textvariable=self.crossover_probability).grid(row=7, column=3, padx=10, pady=5)

        ttk.Label(config_frame, text="Alpha Value (Arithmetic/AlphaMixing/AlphaBetaMixing):").grid(row=7, column=4, padx=10, pady=5)
        self.alpha_value = tk.DoubleVar(value=0.5)
        ttk.Entry(config_frame, textvariable=self.alpha_value).grid(row=7, column=5, padx=10, pady=5)

        ttk.Label(config_frame, text="Beta Value (AlphaBetaMixing):").grid(row=8, column=4, padx=10, pady=5)
        self.beta_value = tk.DoubleVar(value=0.5)
        ttk.Entry(config_frame, textvariable=self.beta_value).grid(row=8, column=5, padx=10, pady=5)

        ttk.Label(config_frame, text="Gaussian Mean (Gaussian):").grid(row=8, column=0, padx=10, pady=5)
        self.gaussian_mean = tk.DoubleVar(value=0)
        ttk.Entry(config_frame, textvariable=self.gaussian_mean).grid(row=8, column=1, padx=10, pady=5)

        ttk.Label(config_frame, text="Gaussian Stddev (Gaussian):").grid(row=8, column=2, padx=10, pady=5)
        self.gaussian_stddev = tk.DoubleVar(value=1)
        ttk.Entry(config_frame, textvariable=self.gaussian_stddev).grid(row=8, column=3, padx=10, pady=5)



        ttk.Label(config_frame, text="Percentage of Best to Transfer (Elitism):").grid(row=9, column=0, padx=10, pady=5)
        self.percentage_best_to_transfer = tk.DoubleVar(value=0.1)
        ttk.Entry(config_frame, textvariable=self.percentage_best_to_transfer).grid(row=9, column=1, padx=10, pady=5)

        ttk.Label(config_frame, text="Fitness Function:").grid(row=10, column=0, padx=10, pady=5)
        self.fitness_function = ttk.Combobox(config_frame, values=["Schwefel", "CECF1"])
        self.fitness_function.grid(row=10, column=1, padx=10, pady=5)
        self.fitness_function.current(0)

        ttk.Label(config_frame, text="Optimization Type:").grid(row=11, column=0, padx=10, pady=5)
        self.optimization_type = ttk.Combobox(config_frame, values=["minimization", "maximization"])
        self.optimization_type.grid(row=11, column=1, padx=10, pady=5)
        self.optimization_type.current(0)

        ttk.Button(config_frame, text="Run Genetic Algorithm", command=self.run_genetic_algorithm).grid(row=12,
                                                                                                        column=0,
                                                                                                        columnspan=3,
                                                                                                        padx=10,
                                                                                                        pady=10)

        self.plot_frame = plot_frame

        self.best_chromosome_label = ttk.Label(main_frame, text="Best Chromosome: N/A")
        self.best_chromosome_label.grid(row=2, column=0, pady=5)

        self.best_fitness_label = ttk.Label(main_frame, text="Best Fitness Value: N/A")
        self.best_fitness_label.grid(row=3, column=0, pady=5)

        self.run_number_label = ttk.Label(main_frame, text="Run Number: 0")
        self.run_number_label.grid(row=4, column=0, pady=5)

        self.computation_time_label = ttk.Label(main_frame, text="Computation Time: N/A")
        self.computation_time_label.grid(row=5, column=0, pady=5)

    def run_genetic_algorithm(self):
        self.run_count += 1
        start_time = time.time()
        num_parameters = self.num_parameters.get()

        gene_range = (self.gene_range_min.get(), self.gene_range_max.get())
        population_size = self.population_size.get()
        num_iterations = self.epochs.get()
        mutation_rate = self.mutation_probability.get()
        crossover_probability = self.crossover_probability.get()

        selection_method_name = self.selection_method.get()
        percentage_best_to_select = self.percentage_best_to_select.get()
        if selection_method_name == "BestSelection":
            selection_method = BestSelection(percentage_the_best_to_select=percentage_best_to_select)
        elif selection_method_name == "TournamentSelection":
            tournament_size = self.tournament_size.get()
            selection_method = TournamentSelection(tournament_size=tournament_size, percentage_the_best_to_select=percentage_best_to_select)
        else:
            selection_method = RouletteWheelSelection(percentage_the_best_to_select=percentage_best_to_select)

        crossover_method_name = self.crossover_method.get()
        if crossover_method_name == "ArithmeticCrossover":
            alpha_value = self.alpha_value.get()
            crossover_method = ArithmeticCrossover(gene_range=gene_range,alpha=alpha_value)
        elif crossover_method_name == "AlphaMixingCrossover":
            alpha_value = self.alpha_value.get()
            crossover_method = AlphaMixingCrossover(gene_range=gene_range, alpha=alpha_value)
        elif crossover_method_name == "AlphaBetaMixingCrossover":
            alpha_value = self.alpha_value.get()
            beta_value = self.beta_value.get()
            crossover_method = AlphaBetaMixingCrossover(gene_range=gene_range, alpha=alpha_value, beta=beta_value)
        elif crossover_method_name == "AveragingCrossover":
            crossover_method = AveragingCrossover(gene_range=gene_range)
        else:
            crossover_method = LinearCrossover(gene_range=gene_range)

        mutation_method_name = self.mutation_method.get()
        if mutation_method_name == "RandomResetting":
            mutation_method = RandomResetting(mutation_rate=mutation_rate, gene_range=gene_range)
        elif mutation_method_name == "UniformMutation":
            mutation_method = UniformMutation(mutation_rate=mutation_rate, gene_range=gene_range)
        elif mutation_method_name == "GaussianMutation":
            mean = self.gaussian_mean.get()
            stddev = self.gaussian_stddev.get()
            mutation_method = GaussianMutation(mutation_rate=mutation_rate, gene_range=gene_range, mean=mean,
                                               stddev=stddev)
        else:
            raise ValueError(f"Unknown mutation method: {mutation_method_name}")

        fitness_function_name = self.fitness_function.get()
        if fitness_function_name == "Schwefel":
            fitness_function = Szwefel()
        else:
            fitness_function = Cec2014F1()

        optimization_type = self.optimization_type.get()
        percentage_best_to_transfer = self.percentage_best_to_transfer.get()



        ga = GeneticAlgorithm(num_parameters=num_parameters, gene_range=gene_range, population_size=population_size,
                              num_iterations=num_iterations,
                              fitness_function=fitness_function, selection_method=selection_method,
                              crossover_method=crossover_method, mutation_method=mutation_method,
                              optimization_type=optimization_type,
                            percentage_best_to_transfer=percentage_best_to_transfer, crossover_probability=crossover_probability)
        best_chromosome, fitness_value_of_best_chromosome, avg_fitness_values, std_fitness_values, min_fitness_values = ga.run()

        rounded_best_chromosome = [round(gene, 3) for gene in best_chromosome.genes]
        rounded_best_fitness_value = round(fitness_value_of_best_chromosome, 3)
        end_time = time.time()
        computation_time = end_time - start_time

        self.plot_fitness(avg_fitness_values, std_fitness_values, min_fitness_values)
        self.best_chromosome_label.config(text=f"Best Chromosome: {rounded_best_chromosome}")
        self.best_fitness_label.config(text=f"Best Fitness Value: {rounded_best_fitness_value}")
        self.run_number_label.config(text=f"Run Number: {self.run_count}")
        self.computation_time_label.config(text=f"Computation Time: {computation_time:.2f} seconds")

        self.save_results(avg_fitness_values, std_fitness_values, min_fitness_values, rounded_best_chromosome,
                          rounded_best_fitness_value, computation_time)

    def plot_fitness(self, avg_fitness_values, std_fitness_values, min_fitness_values):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        max_points = 500
        iterations = range(1, len(avg_fitness_values) + 1)

        if len(iterations) > max_points:
            sampled_indices = np.linspace(0, len(iterations) - 1, max_points, dtype=int)
            iterations = np.array(iterations)[sampled_indices]
            avg_fitness_values = np.array(avg_fitness_values)[sampled_indices]
            std_fitness_values = np.array(std_fitness_values)[sampled_indices]
            min_fitness_values = np.array(min_fitness_values)[sampled_indices]

        fig = Figure(figsize=(18, 6))

        ax1 = fig.add_subplot(131)
        ax1.plot(iterations, avg_fitness_values, label='Average Fitness')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Average Fitness Value')
        ax1.set_title('Average Fitness Value per Iteration')
        ax1.legend()
        ax1.text(0.5, 0.9, f"Run: {len(avg_fitness_values)}", transform=ax1.transAxes, ha="center")

        ax2 = fig.add_subplot(132)
        ax2.plot(iterations, std_fitness_values, label='Standard Deviation of Fitness', color='orange')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Standard Deviation of Fitness per Iteration')
        ax2.legend()
        ax2.text(0.5, 0.9, f"Run: {len(std_fitness_values)}", transform=ax2.transAxes, ha="center")

        ax3 = fig.add_subplot(133)
        ax3.plot(iterations, min_fitness_values, label='Minimum Fitness', color='green')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Minimum Fitness Value')
        ax3.set_title('Minimum Fitness Value per Iteration')
        ax3.legend()
        ax3.text(0.5, 0.9, f"Run: {len(min_fitness_values)}", transform=ax3.transAxes, ha="center")

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_results(self, avg_fitness_values, std_fitness_values, min_fitness_values, best_chromosome, best_fitness_value, computation_time):
        run_dir = f"C:/Users/Konrad/PycharmProjects/real-value-based-chromosome/app/results/{self.first_run_date}/run_{self.run_count}"
        os.makedirs(run_dir, exist_ok=True)

        self.save_to_csv(os.path.join(run_dir, "avg_fitness_values.csv"), avg_fitness_values)
        self.save_to_csv(os.path.join(run_dir, "std_fitness_values.csv"), std_fitness_values)
        self.save_to_csv(os.path.join(run_dir, "min_fitness_values.csv"), min_fitness_values)
        self.save_plots(run_dir, avg_fitness_values, std_fitness_values, min_fitness_values)
        self.save_model_params_and_results(run_dir, best_chromosome, best_fitness_value, computation_time)

    def save_plots(self, run_dir, avg_fitness_values, std_fitness_values, min_fitness_values):
        max_points = 500
        iterations = range(1, len(avg_fitness_values) + 1)

        if len(iterations) > max_points:
            sampled_indices = np.linspace(0, len(iterations) - 1, max_points, dtype=int)
            iterations = np.array(iterations)[sampled_indices]
            avg_fitness_values = np.array(avg_fitness_values)[sampled_indices]
            std_fitness_values = np.array(std_fitness_values)[sampled_indices]
            min_fitness_values = np.array(min_fitness_values)[sampled_indices]

        fig = Figure(figsize=(18, 6))

        ax1 = fig.add_subplot(131)
        ax1.plot(iterations, avg_fitness_values, label='Average Fitness')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Average Fitness Value')
        ax1.set_title('Average Fitness Value per Iteration')
        ax1.legend()

        ax2 = fig.add_subplot(132)
        ax2.plot(iterations, std_fitness_values, label='Standard Deviation of Fitness', color='orange')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Standard Deviation of Fitness per Iteration')
        ax2.legend()

        ax3 = fig.add_subplot(133)
        ax3.plot(iterations, min_fitness_values, label='Minimum Fitness', color='green')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Minimum Fitness Value')
        ax3.set_title('Minimum Fitness Value per Iteration')
        ax3.legend()

        fig.savefig(os.path.join(run_dir, "fitness_plots.png"))

    def save_model_params_and_results(self, run_dir, best_chromosome, best_fitness_value, computation_time):
        params_file = os.path.join(run_dir, "model_params_and_results.txt")
        with open(params_file, 'w') as file:
            file.write(f"Number of Parameters: {self.num_parameters.get()}\n")
            file.write(f"Gene Range: ({self.gene_range_min.get()}, {self.gene_range_max.get()})\n")
            file.write(f"Population Size: {self.population_size.get()}\n")
            file.write(f"Epochs: {self.epochs.get()}\n")
            file.write(f"Mutation Probability: {self.mutation_probability.get()}\n")
            file.write(f"Fitness Function: {self.fitness_function.get()}\n")
            file.write(f"Optimization Type: {self.optimization_type.get()}\n")
            file.write(f"Selection Method: {self.selection_method.get()}\n")
            file.write(f"Percentage of Best to Select: {self.percentage_best_to_select.get()}\n")
            file.write(f"Eitism Percentage: {self.percentage_best_to_transfer.get()}\n")
            file.write(f"Crossover Probability: {self.crossover_probability.get()}\n")


            if self.selection_method.get() == "TournamentSelection":
                file.write(f"Tournament Size: {self.tournament_size.get()}\n")

            if self.crossover_method.get() in ["AlphaBetaMixingCrossover", "AlphaMixingCrossover",
                                               "ArithmeticCrossover"]:
                file.write(f"Alpha Value: {self.alpha_value.get()}\n")

            if self.crossover_method.get() == "AlphaBetaMixingCrossover":
                file.write(f"Beta Value: {self.beta_value.get()}\n")

            file.write(f"Crossover Method: {self.crossover_method.get()}\n")
            file.write(f"Mutation Method: {self.mutation_method.get()}\n")

            if self.mutation_method.get() == "GaussianMutation":
                file.write(f"Gaussian Mean: {self.gaussian_mean.get()}\n")
                file.write(f"Gaussian Stddev: {self.gaussian_stddev.get()}\n")

            file.write(f"Best Chromosome: {best_chromosome}\n")
            file.write(f"Best Fitness Value: {best_fitness_value}\n")
            file.write(f"Computation Time: {computation_time} seconds\n")

    def save_to_csv(self, file_path, data):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Iteration", "Value"])
            for i, value in enumerate(data, start=1):
                writer.writerow([i, value])


if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmApp(root)
    root.mainloop()
    def plot_fitness(self, avg_fitness_values, std_fitness_values, min_fitness_values):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        max_points = 500
        iterations = range(1, len(avg_fitness_values) + 1)

        if len(iterations) > max_points:
            sampled_indices = np.linspace(0, len(iterations) - 1, max_points, dtype=int)
            iterations = np.array(iterations)[sampled_indices]
            avg_fitness_values = np.array(avg_fitness_values)[sampled_indices]
            std_fitness_values = np.array(std_fitness_values)[sampled_indices]
            min_fitness_values = np.array(min_fitness_values)[sampled_indices]

        fig = Figure(figsize=(18, 6))

        ax1 = fig.add_subplot(131)
        ax1.plot(iterations, avg_fitness_values, label='Average Fitness')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Average Fitness Value')
        ax1.set_title('Average Fitness Value per Iteration')
        ax1.legend()
        ax1.text(0.5, 0.9, f"Run: {len(avg_fitness_values)}", transform=ax1.transAxes, ha="center")

        ax2 = fig.add_subplot(132)
        ax2.plot(iterations, std_fitness_values, label='Standard Deviation of Fitness', color='orange')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Standard Deviation of Fitness per Iteration')
        ax2.legend()
        ax2.text(0.5, 0.9, f"Run: {len(std_fitness_values)}", transform=ax2.transAxes, ha="center")

        ax3 = fig.add_subplot(133)
        ax3.plot(iterations, min_fitness_values, label='Minimum Fitness', color='green')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Minimum Fitness Value')
        ax3.set_title('Minimum Fitness Value per Iteration')
        ax3.legend()
        ax3.text(0.5, 0.9, f"Run: {len(min_fitness_values)}", transform=ax3.transAxes, ha="center")

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_results(self, avg_fitness_values, std_fitness_values, min_fitness_values, best_chromosome, best_fitness_value, computation_time):
        run_dir = f"C:/Users/Konrad/PycharmProjects/real-value-based-chromosome/app/results/{self.first_run_date}/run_{self.run_count}"
        os.makedirs(run_dir, exist_ok=True)

        self.save_to_csv(os.path.join(run_dir, "avg_fitness_values.csv"), avg_fitness_values)
        self.save_to_csv(os.path.join(run_dir, "std_fitness_values.csv"), std_fitness_values)
        self.save_to_csv(os.path.join(run_dir, "min_fitness_values.csv"), min_fitness_values)
        self.save_plots(run_dir, avg_fitness_values, std_fitness_values, min_fitness_values)
        self.save_model_params_and_results(run_dir, best_chromosome, best_fitness_value, computation_time)

    def save_plots(self, run_dir, avg_fitness_values, std_fitness_values, min_fitness_values):
        max_points = 500
        iterations = range(1, len(avg_fitness_values) + 1)

        if len(iterations) > max_points:
            sampled_indices = np.linspace(0, len(iterations) - 1, max_points, dtype=int)
            iterations = np.array(iterations)[sampled_indices]
            avg_fitness_values = np.array(avg_fitness_values)[sampled_indices]
            std_fitness_values = np.array(std_fitness_values)[sampled_indices]
            min_fitness_values = np.array(min_fitness_values)[sampled_indices]

        fig = Figure(figsize=(18, 6))

        ax1 = fig.add_subplot(131)
        ax1.plot(iterations, avg_fitness_values, label='Average Fitness')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Average Fitness Value')
        ax1.set_title('Average Fitness Value per Iteration')
        ax1.legend()

        ax2 = fig.add_subplot(132)
        ax2.plot(iterations, std_fitness_values, label='Standard Deviation of Fitness', color='orange')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Standard Deviation of Fitness per Iteration')
        ax2.legend()

        ax3 = fig.add_subplot(133)
        ax3.plot(iterations, min_fitness_values, label='Minimum Fitness', color='green')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Minimum Fitness Value')
        ax3.set_title('Minimum Fitness Value per Iteration')
        ax3.legend()

        fig.savefig(os.path.join(run_dir, "fitness_plots.png"))


    def save_model_params_and_results(self, run_dir, best_chromosome, best_fitness_value, computation_time):
        params_file = os.path.join(run_dir, "model_params_and_results.txt")
        with open(params_file, 'w') as file:
            file.write(f"Number of Parameters: {self.num_parameters.get()}\n")
            file.write(f"Gene Range: ({self.gene_range_min.get()}, {self.gene_range_max.get()})\n")
            file.write(f"Population Size: {self.population_size.get()}\n")
            file.write(f"Epochs: {self.epochs.get()}\n")
            file.write(f"Mutation Rate: {self.mutation_probability.get()}\n")
            file.write(f"Fitness Function: {self.fitness_function.get()}\n")
            file.write(f"Optimization Type: {self.optimization_type.get()}\n")
            file.write(f"Selection Method: {self.selection_method.get()}\n")
            file.write(f"Percentage of Best to Select: {self.percentage_best_to_select.get()}\n")
            if self.selection_method.get() == "TournamentSelection":
                file.write(f"Tournament Size: {self.tournament_size.get()}\n")
            if self.crossover_method.get() == "AlphaBetaMixingCrossover":
                file.write(f"Beta Value: {self.beta_value.get()}\n")

            file.write(f"Alpha Value: {self.alpha_value.get()}\n")
            file.write(f"Crossover Method: {self.crossover_method.get()}\n")  # Save the crossover method name
            file.write(f"Best Chromosome: {best_chromosome}\n")
            file.write(f"Best Fitness Value: {best_fitness_value}\n")
            file.write(f"Computation Time: {computation_time} seconds\n")

    def save_to_csv(self, file_path, data):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Iteration", "Value"])
            for i, value in enumerate(data, start=1):
                writer.writerow([i, value])


if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticAlgorithmApp(root)
    root.mainloop()