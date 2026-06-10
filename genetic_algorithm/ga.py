import numpy as np


def tournament_select(population, fitnesses, k=3):
    # randomly pick k individuals and return the best one
    # k=3 works well in practice — enough pressure without losing diversity
    idx = np.random.choice(len(population), k, replace=False)
    best = idx[np.argmax([fitnesses[i] for i in idx])]
    return population[best].copy()



def crossover(parent1, parent2):
    # cut both parents at the same random point and swap the tails
    point = np.random.randint(1, len(parent1))
    child = np.concatenate([parent1[:point], parent2[point:]])
    return child


def mutate(individual, mutation_rate, w_min=0.0, w_max=1.0):
    # randomly reset some genes to keep the search from getting stuck
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.uniform(w_min, w_max)
    return individual


def run_ga(fitness_fn, n_weights, pop_size=20, n_generations=30,
           mutation_rate=0.1, w_min=0.0, w_max=1.0, seed=None):

    if seed is not None:
        np.random.seed(seed)

    population = [np.random.uniform(w_min, w_max, n_weights)
                  for _ in range(pop_size)]

    best_fitness_history = []
    avg_fitness_history  = []
    best_individual      = None
    best_fitness         = -np.inf

    for gen in range(n_generations):
        fitnesses = [fitness_fn(ind) for ind in population]

        gen_best = max(fitnesses)
        gen_avg  = np.mean(fitnesses)
        best_fitness_history.append(gen_best)
        avg_fitness_history.append(gen_avg)

        if gen_best > best_fitness:
            best_fitness    = gen_best
            best_individual = population[np.argmax(fitnesses)].copy()

        print(f'Gen {gen+1:3d} | best={gen_best:.6f} | avg={gen_avg:.6f}')

        # elitism — best individual goes straight into the next generation
        next_pop = [best_individual.copy()]

        while len(next_pop) < pop_size:
            p1    = tournament_select(population, fitnesses)
            p2    = tournament_select(population, fitnesses)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate, w_min, w_max)
            next_pop.append(child)

        population = next_pop

    return best_individual, best_fitness, best_fitness_history, avg_fitness_history

if __name__ == '__main__':
    # testing on Rastrigin before connecting to FEMM
    # it has many local minima which makes it a good GA benchmark
    # global minimum is at x=0 for all genes, which maps to w=0.5 after our shift

    def rastrigin(w):
        x   = (w - 0.5) * 10
        val = 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        return -val  # negate so GA maximizes toward 0

    best_w, best_f, hist_best, hist_avg = run_ga(
        fitness_fn    = rastrigin,
        n_weights     = 16,
        pop_size      = 30,
        n_generations = 50,
        mutation_rate = 0.1,
        seed          = 42
    )

    print(f'\nBest fitness: {best_f:.4f}')
    print(f'Best weights (first 5): {best_w[:5].round(4)}')