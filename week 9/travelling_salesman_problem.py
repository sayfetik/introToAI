import numpy as np
import matplotlib.pyplot as plt
import imageio

# Initial data: coordinates of cities
cities = np.array([[35, 51],
                   [113, 213],
                   [82, 280],
                   [322, 340],
                   [256, 352],
                   [160, 24],
                   [322, 145],
                   [12, 349],
                   [282, 20],
                   [241, 8],
                   [398, 153],
                   [182, 305],
                   [153, 257],
                   [275, 190],
                   [242, 75],
                   [19, 229],
                   [303, 352],
                   [39, 309],
                   [383, 79],
                   [226, 343]])


def compute_distances(cities):
    distances = []
    for from_city in cities:
        row = []
        for to_city in cities:
            row.append(np.linalg.norm(from_city - to_city))
        distances.append(row)
    return np.array(distances)


def create_fitness(distances):
    # Returns a fitness function for evaluating routes
    def fitness(individual):
        res = 0
        for i in range(len(individual)):
            # Add distance from current city to next city
            res += distances[individual[i], individual[(i + 1) % len(individual)]]
        # Return the negative of the total distance
        return -res

    return fitness


def route_length(distances, route):
    length = 0
    for i in range(len(route)):
        length += distances[route[i], route[(i + 1) % len(route)]]
    return length


def plot_route(cities, route, distances, generation, show=False):
    length = route_length(distances, route)

    plt.figure(figsize=(12, 8))
    plt.scatter(x=cities[:, 0], y=cities[:, 1], s=1000, zorder=1)
    for i in range(len(cities)):
        plt.text(cities[i][0], cities[i][1], str(i), horizontalalignment='center', verticalalignment='center', size=16,
                 c='white')
    for i in range(len(route)):
        plt.plot([cities[route[i]][0], cities[route[(i + 1) % len(route)]][0]],
                 [cities[route[i]][1], cities[route[(i + 1) % len(route)]][1]], 'k', zorder=0)

    plt.title(f'Generation: {generation}. Visiting {len(route)} cities in length {length:.2f}', size=16)

    if show:
        plt.show()

    plt.savefig(f'{generation}.png')
    plt.close()


distances = compute_distances(cities)
fitness = create_fitness(distances)
plot_route(cities, route=[], distances=distances, generation=-1, show=True)


## Implementation

def get_individual(n_cities):
    # Returns one individual (route) from the population.
    return np.random.permutation(n_cities)


def initial_population(n_cities, population_size, fitness):
    population = [get_individual(n_cities) for _ in range(population_size)]
    population.sort(key=lambda x: fitness(x))
    return population


def get_parents(population, n_offsprings):
    mothers = population[-2 * n_offsprings::2]
    fathers = population[-2 * n_offsprings + 1::2]
    return mothers, fathers


def cross(mother, father):
    mother_head = mother[:int(len(mother) * 0.5)].copy()
    mother_tail = mother[int(len(mother) * 0.5):].copy()
    father_tail = father[int(len(father) * 0.5):].copy()

    # Create a mapping from cities in the father_tail to cities in the mother_tail.
    mapping = {father[i]: mother_tail[i] for i in range(len(father_tail))}

    # Replace the cities in mother_head with their mapped values if they exist in father_tail.
    for i in range(len(mother_head)):
        if mother_head[i] in mapping:
            mother_head[i] = mapping[mother_head[i]]

    # Combine the modified mother_head and father_tail to form the new offspring.
    offspring = np.concatenate((mother_head, father_tail))

    return offspring


def mutate(offspring):
    # Mutate the given offspring by swapping two cities.
    indices = np.random.choice(len(offspring), size=2, replace=False)
    offspring[indices[0]], offspring[indices[1]] = offspring[indices[1]], offspring[indices[0]]
    return offspring


def replace_population(population, new_individuals, fitness):
    # Add new individuals to the current population and sort based on fitness.
    population.extend(new_individuals)
    population.sort(key=lambda x: fitness(x))

    # Return only the top individuals that make up the original population size.
    return population[-len(new_individuals):]


def evolution_step(population, population_size, fitness, n_offsprings):
    mothers, fathers = get_parents(population, n_offsprings)

    offsprings = []

    for mother, father in zip(mothers, fathers):
        offspring = mutate(cross(mother, father))
        offsprings.append(offspring)

    new_population = replace_population(population.copy(), offsprings, fitness)

    return new_population


def evolution(n_cities, fitness_function, population_size=100, n_offsprings=30, generations=10):
    fitness_change = []

    population = initial_population(n_cities, population_size, fitness_function)

    for generation in range(generations):
        population = evolution_step(population.copy(), population_size, fitness_function, n_offsprings)
        best_individual = population[-1]
        fitness_change.append(fitness_function(best_individual))
        plot_route(cities, route=best_individual.tolist(), distances=distances,
                   generation=generation)

    return fitness_change


generations = 500
fitness_change = evolution(len(cities), create_fitness(distances),
                           population_size=100,
                           n_offsprings=30,
                           generations=generations)

plt.plot(fitness_change)
plt.title('Change of a Fitness Score')
plt.xlabel('Generation')
plt.ylabel('Fitness Score')
plt.show()

from IPython.display import Image

with imageio.get_writer('mygif.gif', mode='I') as writer:
    for generation in range(generations):
        image = imageio.imread(f'{generation}.png')
        writer.append_data(image)

Image(open('mygif.gif', 'rb').read())
Image(open(f'{generations - 1}.png', 'rb').read())
