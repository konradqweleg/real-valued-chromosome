import os
import csv
from datetime import datetime
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from app.crossover.arithmetic_crossover import ArithmeticCrossover
from app.function.szwefel import Szwefel
from app.mutation.random_resetting import RandomResetting
from app.selection.best_selection import BestSelection
from app.genetic_algorithm.genetic_algorithm import GeneticAlgorithm


class GeneticAlgorithmApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Genetic Algorithm Configuration")
        self.run_count = 0
        self.first_run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.create_widgets()

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

        ttk.Label(config_frame, text="Number of Iterations:").grid(row=3, column=0, padx=10, pady=5)
        self.num_iterations = tk.IntVar(value=10000)
        ttk.Entry(config_frame, textvariable=self.num_iterations).grid(row=3, column=1, padx=10, pady=5)

        ttk.Label(config_frame, text="Mutation Rate:").grid(row=4, column=0, padx=10, pady=5)
        self.mutation_rate = tk.DoubleVar(value=0.01)
        ttk.Entry(config_frame, textvariable=self.mutation_rate).grid(row=4, column=1, padx=10, pady=5)

        ttk.Label(config_frame, text="Selection Method:").grid(row=5, column=0, padx=10, pady=5)
        self.selection_method = ttk.Combobox(config_frame, values=["BestSelection"])
        self.selection_method.grid(row=5, column=1, padx=10, pady=5)
        self.selection_method.current(0)

        ttk.Label(config_frame, text="Crossover Method:").grid(row=6, column=0, padx=10, pady=5)
        self.crossover_method = ttk.Combobox(config_frame, values=["ArithmeticCrossover"])
        self.crossover_method.grid(row=6, column=1, padx=10, pady=5)
        self.crossover_method.current(0)

        ttk.Label(config_frame, text="Mutation Method:").grid(row=7, column=0, padx=10, pady=5)
        self.mutation_method = ttk.Combobox(config_frame, values=["RandomResetting"])
        self.mutation_method.grid(row=7, column=1, padx=10, pady=5)
        self.mutation_method.current(0)

        ttk.Button(config_frame, text="Run Genetic Algorithm", command=self.run_genetic_algorithm).grid(row=8, column=0, columnspan=3, padx=10, pady=10)

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
        num_iterations = self.num_iterations.get()
        mutation_rate = self.mutation_rate.get()

        selection_method = BestSelection(percentage_the_best_to_select=0.5)
        crossover_method = ArithmeticCrossover(alpha=0.5, gene_range=gene_range)
        mutation_method = RandomResetting(mutation_rate=mutation_rate, gene_range=gene_range)

        ga = GeneticAlgorithm(num_parameters=num_parameters, gene_range=gene_range, population_size=population_size, num_iterations=num_iterations,
                              fitness_function=Szwefel(), selection_method=selection_method, crossover_method=crossover_method, mutation_method=mutation_method, optimization_type='minimization')
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

        self.save_results(avg_fitness_values, std_fitness_values, min_fitness_values, rounded_best_chromosome, rounded_best_fitness_value, computation_time)

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
            file.write(f"Number of Iterations: {self.num_iterations.get()}\n")
            file.write(f"Mutation Rate: {self.mutation_rate.get()}\n")
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