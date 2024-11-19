#include <iostream>
#include <utility>
#include <vector>
#include <random>
#include <set>
#include <fstream>
#include <type_traits>
#include <algorithm>
#include <chrono>

using namespace std;

// Constant for grid size and minimum value representation
const int N = 9;
const int MINUS_INFINITY = numeric_limits<int>::min();

// Maximum number of generations for the genetic algorithm
int MAX_GENERATION = 100000;

// File paths for input and output
string INPUT_FILE = "C:/didi/Inno/2 year 1 semester/introToAI/sudoku_ga/input.txt";
string OUTPUT_FILE = "C:/didi/Inno/2 year 1 semester/introToAI/sudoku_ga/output.txt";

// Enum to represent types of mutation operations
enum Mutation {
    ROW, COLUMN, SQUARE
};

// Template function to generate a random number in the specified range
template<typename T>
T getRandom(T start, T end) {
    mt19937 generator(random_device{}()); // Random number generator

    if constexpr (std::is_floating_point<T>::value) {
        uniform_real_distribution<T> distribution(start, end); // Distribution for floating point numbers
        return distribution(generator);
    } else if constexpr (std::is_integral<T>::value) {
        uniform_int_distribution<T> distribution(start, end); // Distribution for integral values
        return distribution(generator);
    }
}

// Generate a random mutation type (ROW, COLUMN, or SQUARE)
Mutation getRandomMutation() {
    random_device rd;
    mt19937 generator(rd());
    uniform_int_distribution<int> distribution(0, 2); // Range: 0-2 for mutation types
    return static_cast<Mutation>(distribution(generator));
}

// Generate a random digit from 1 to N
int getRandomDigit() {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, N); // Range: 1 to N
    return distrib(gen);
}

// Class to represent a Sudoku table
class Table {
private:
    vector<vector<int>> grid; // Sudoku grid
    int fitness = 0; // Fitness value for the table

public:
    // Default constructor to initialize an empty grid
    Table() : grid(N, vector<int>(N, 0)) {
        this->setFitness();
    }

    // Constructor with provided grid
    explicit Table(vector<vector<int>> g) : grid(std::move(g)) {
        this->setFitness();
    }

    // Getter for the grid
    vector<vector<int>> getGrid() {
        return this->grid;
    }

    // Get possible digits for a given cell (row, col)
    vector<int> getPossible(int row, int col) {
        set<int> possible_digits = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        // Check row and remove existing digits
        for (int j = 0; j < N; ++j) {
            if (this->grid[row][j] != 0) possible_digits.erase(this->grid[row][j]);
        }
        // Check column and remove existing digits
        for (int i = 0; i < N; ++i) {
            if (this->grid[i][col] != 0) possible_digits.erase(this->grid[i][col]);
        }
        // Check 3x3 square and remove existing digits
        int square_row = (row / 3) * 3;
        int square_col = (col / 3) * 3;
        for (int i = square_row; i < square_row + 3; ++i) {
            for (int j = square_col; j < square_col + 3; ++j) {
                if (this->grid[i][j] != 0) possible_digits.erase(this->grid[i][j]);
            }
        }
        // Convert set to vector and return
        vector<int> result;
        for (int d: possible_digits) {
            result.push_back(d);
        }
        return result;
    }

    // Fill cells with predetermined values if possible
    void predeterminedCells() {
        while (true) {
            bool is_updated = false;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    if (this->grid[i][j] == 0) {
                        vector<int> possible_digits = this->getPossible(i, j);
                        if (possible_digits.size() == 1) {
                            this->grid[i][j] = possible_digits[0];
                            is_updated = true;
                        }
                    }
                }
            }
            if (!is_updated) {
//                cout << "No more obvious squares\n"; // Debug message (commented out)
                break;
            }
        }
    }

    // Generate a random solution based on the current grid state
    Table get_random_solution() {
        vector<vector<int>> solution = this->grid;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (grid[i][j] == 0) {
                    vector<int> possible_digits = this->getPossible(i, j);
                    if (!possible_digits.empty()) solution[i][j] = possible_digits[getRandom(0, int(possible_digits.size()) - 1)];
                    else solution[i][j] = getRandomDigit(); // Random digit fallback
                }
            }
        }
        return Table(solution);
    }

    // Overload for printing the Table to an output stream
    friend ostream &operator<<(ostream &out, const Table &t) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                out << t.grid[i][j] << " ";
            }
            out << endl;
        }
        return out;
    }

    // Overload for reading a Table from an input stream
    friend istream &operator>>(istream &in, Table &t) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                char c;
                in >> c;
                if (c != '-' && c != '0') {
                    t.grid[i][j] = c - '0'; // Convert character to integer
                } else t.grid[i][j] = 0;
            }
        }
        return in;
    }

    // Retrieve digits in each row
    vector<set<int>> rowDigits() {
        vector<set<int>> res(9);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                res[i].insert(this->grid[i][j]);
            }
        }
        return res;
    }

    // Retrieve digits in each column
    vector<set<int>> columnDigits() {
        vector<set<int>> res(9);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                res[j].insert(this->grid[i][j]);
            }
        }
        return res;
    }

    // Retrieve digits in each 3x3 square
    vector<set<int>> squareDigits() {
        vector<set<int>> res(9);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int squareIndex = (i / 3) * 3 + (j / 3);
                res[squareIndex].insert(grid[i][j]);
            }
        }
        return res;
    }

    // Calculate and set the fitness value for the grid
    void setFitness() {
        int res = 0;
        vector<set<int>> rowDigits = this->rowDigits();
        vector<set<int>> columnDigits = this->columnDigits();
        vector<set<int>> squareDigits = this->squareDigits();
        for (int i = 0; i < N; ++i) {
            res += int(rowDigits[i].size()) + int(columnDigits[i].size()) + int(squareDigits[i].size());
        }
        this->fitness = res;
    }

    // Getter for fitness value
    int getFitness() const {
        return this->fitness;
    }

    // Check if the grid is solved
    bool is_solved() const {
        return this->fitness == 243; // Maximum fitness value when all rows, columns, and squares are filled correctly
    }
};

// Read the Sudoku table from a file
bool readTableFromFile(Table& s) {
    ifstream inputFile(INPUT_FILE);
    if (inputFile.is_open()) {
        inputFile >> s;
        inputFile.close();
        return true;
    } else {
        cerr << "Error opening input.txt!" << endl;
        return false;
    }
}

// Write the Sudoku table to a file
bool writeToFile(Table& content) {
    ofstream outFile(OUTPUT_FILE);
    if (outFile.is_open()) {
        outFile << content;
        outFile.close();
        return true;
    } else {
        cout << "Error opening output.txt!" << endl;
        return false;
    }
}

class GeneticAlgorithm {
private:
    // Parameters for genetic algorithm
    int population_size = 700;
    double mutation_rate = 0.04;
    double mutation_rate_factor = 1.5;
    int stagnation_threshold = 50;
    int random_sample_size = 10;
    double random_selection_portion = 0.1;
    double elitism_portion = 0.1;
    int best_fitness = -1;
    int stagnant_generations = 0;
    double max_mutation_rate = 0.05;
    double min_mutation_rate = 0.01;
    Table table;
public:
    // Constructor that initializes the genetic algorithm with a table
    explicit GeneticAlgorithm(Table t) {
        t.predeterminedCells(); // Mark predetermined cells in the table
        this->table = t;
        this->table.setFitness(); // Set initial fitness value of the table
    }

    // Getter for the table object
    Table getTable() {
        return this->table;
    }

    // Generate initial population for the genetic algorithm
    vector<Table> get_population() {
        vector<Table> population;
        for (int i = 0; i < this->population_size; ++i) {
            population.push_back(this->table.get_random_solution());
        }
        return population;
    }

    // Mutate a table using a random mutation type
    Table mutate(Table &t) const {
        vector<vector<int>> mutated_solution = t.getGrid();
        for (int i = 0; i < N; ++i) {
            if (getRandom(0.0, 1.0) < this->mutation_rate) {
                Mutation mutation_type = getRandomMutation();
                switch (mutation_type) {
                    case ROW:
                        // Swap two elements in a row
                        swap(mutated_solution[i][getRandom(0, N - 1)], mutated_solution[i][getRandom(0, N - 1)]);
                        break;
                    case COLUMN:
                        // Swap two elements in a column
                        swap(mutated_solution[getRandom(0, N - 1)][i], mutated_solution[getRandom(0, N - 1)][i]);
                        break;
                    case SQUARE:
                        // Perform mutation in a 3x3 square
                        int square_row = (i / 3) * 3;
                        int square_col = (i % 3) * 3;
                        vector<pair<int, int>> square_cells;
                        for (int j = square_row; j < square_row + 3; ++j) {
                            for (int k = square_col; k < square_col + 3; ++k) {
                                if (j < N && k < N) square_cells.emplace_back(j, k);
                            }
                        }
                        int size_square_cells = int(square_cells.size()) - 1;
                        auto first_random_cell = square_cells[getRandom(0, size_square_cells)];
                        auto second_random_cell = square_cells[getRandom(0, size_square_cells)];
                        int i1 = first_random_cell.first;
                        int j1 = first_random_cell.second;
                        int i2 = second_random_cell.first;
                        int j2 = second_random_cell.second;
                        // Swap two elements within the square
                        swap(mutated_solution[i1][j1], mutated_solution[i2][j2]);
                        break;
                }
            }
        }
        return Table(mutated_solution);
    }

    // Perform crossover operation between two parent tables
    static Table crossover(Table parent1, Table parent2) {
        int crossover_point = getRandomDigit(); // Random crossover point
        vector<vector<int>> child;
        // Inherit rows from parent1 up to the crossover point
        for (int i = 0; i < crossover_point; ++i) {
            child.push_back(parent1.getGrid()[i]);
        }
        // Inherit remaining rows from parent2
        for (int i = crossover_point; i < N; ++i) {
            child.push_back(parent2.getGrid()[i]);
        }
        return Table(child);
    }

    // Check if any solution in the population is solved
    static Table *get_solution(vector<Table> population) {
        int n = int(population.size());
        for (int i = 0; i < n; ++i) {
            if (population[i].is_solved()) return &population[i];
        }
        return nullptr;
    }

    // Get the best solution from a given population
    static Table get_best_solution(vector<Table> &population) {
        int largest_fitness = MINUS_INFINITY;
        Table best_solution;
        for (const auto &solution: population) {
            if (solution.getFitness() > largest_fitness) {
                largest_fitness = solution.getFitness();
                best_solution = solution;
            }
        }
        return best_solution;
    }

    // Choose a parent for reproduction using random sampling and tournament selection
    Table choose_parent(vector<Table> population) const {
        vector<Table> random_sample;
        int tables_number = int(population.size()) - 1;
        for (int i = 0; i < this->random_sample_size; ++i) {
            random_sample.push_back(population[getRandom(0, tables_number)]);
        }
        return get_best_solution(random_sample); // Return the best solution among the sample
    }

    // Perform a generation of genetic algorithm evolution
    vector<Table> genetic_algorithm(vector<Table> population) {
        vector<Table> new_population;
        // Perform elitism by copying a portion of the best individuals to the new population
        int num_elitism = int(int(population.size()) * this->elitism_portion);
        vector<Table> sorted_population = population;
        sort(sorted_population.begin(), sorted_population.end(),
             [](Table &a, Table &b) { return a.getFitness() > b.getFitness(); });
        for (int i = 0; i < num_elitism; ++i) {
            new_population.push_back(sorted_population[i]);
        }

        // Random selection for maintaining diversity
        int num_random_selection = int((int(population.size()) - num_elitism) * this->random_selection_portion);
        int num_tournament_selection = int(population.size()) - num_elitism - num_random_selection;

        for (int i = 0; i < num_random_selection; ++i) {
            new_population.push_back(population[i]);
        }

        // Perform crossover to create new individuals
        for (int i = 0; i < num_tournament_selection; ++i) {
            Table parent1 = this->choose_parent(population);
            Table parent2 = this->choose_parent(population);
            Table child = GeneticAlgorithm::crossover(parent1, parent2);
            new_population.push_back(child);
        }

        // Apply mutation to the new population
        int new_population_size = int(new_population.size());
        for (int i = 0; i < new_population_size; ++i) {
            new_population[i] = this->mutate(new_population[i]);
        }

        // Update fitness and mutation rate based on progress
        auto cur_best_solution = GeneticAlgorithm::get_best_solution(new_population);
        int cur_best_fitness = cur_best_solution.getFitness();

        // Adjust mutation rate based on progress
        if (cur_best_fitness > this->best_fitness) {
            this->best_fitness = cur_best_fitness;
            this->stagnant_generations = 0;
            this->mutation_rate *= (1 / this->mutation_rate_factor);
            this->mutation_rate = max(this->mutation_rate, this->min_mutation_rate);
        } else this->stagnant_generations += 1;

        // Increase mutation rate if stagnation occurs
        if (this->stagnant_generations > this->stagnation_threshold) {
            this->mutation_rate *= this->mutation_rate_factor;
            this->mutation_rate = min(this->mutation_rate, this->max_mutation_rate);
            this->stagnant_generations = 0;
        }

        // Return the new population if no solution is found
        if (cur_best_solution.is_solved()) {
            vector<Table> answer = {cur_best_solution};
            return answer;
        }
        return new_population;
    }
};

// Function for solving sudoku
void solve_sudoku(Table t) {
    // Initialize the genetic algorithm solver with the provided table
    GeneticAlgorithm sudoku_solver(std::move(t));

    // Check if the initial table is already solved
    if (sudoku_solver.getTable().getFitness() == 243) {
        cout << sudoku_solver.getTable(); // Print the solved table
        return; // Exit function
    }

    // Generate initial population and evolve through generations
    vector<Table> population = sudoku_solver.get_population();

    for (int generation = 0; generation < MAX_GENERATION; ++generation) {
        // Get the best solution in the current generation
        Table best_solution = GeneticAlgorithm::get_best_solution(population);
        int best_fitness = best_solution.getFitness();
        cout << "Generation" << generation + 1 << ": Best Fitness: " << best_fitness << "/243" << endl;

        // Perform the genetic algorithm evolution
        population = sudoku_solver.genetic_algorithm(population);

        // Check if a solution has been found
        Table *solution = GeneticAlgorithm::get_solution(population);
        if (solution != nullptr) {
            writeToFile(*solution); // Write the solution to a file
            return;
        }
    }

    // If no solution is found within the generation limit, print the best attempt
    cout << "Solution not found!" << endl;
}

// Main function
int main() {
    auto start = chrono::high_resolution_clock::now();

    Table s;
    if (!readTableFromFile(s)) return 1;
    solve_sudoku(s);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Execution time: " << elapsed.count() << " seconds" << endl;

    return 0;
}
