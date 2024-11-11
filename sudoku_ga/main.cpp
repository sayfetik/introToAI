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

const int N = 9;
const int MINUS_INFINITY = numeric_limits<int>::min();
int MAX_GENERATION = 100000;

enum Mutation {
    ROW, COLUMN, SQUARE
};

template <typename T>
T getRandom(T start, T end) {
    mt19937 generator(random_device{}());

    if constexpr (std::is_floating_point<T>::value) {
        uniform_real_distribution<T> distribution(start, end);
        return distribution(generator);
    }
    else if constexpr (std::is_integral<T>::value) {
        uniform_int_distribution<T> distribution(start, end);
        return distribution(generator);
    }
}

Mutation getRandomMutation() {
    random_device rd;
    mt19937 generator(rd());
    uniform_int_distribution<int> distribution(0, 2);
    return static_cast<Mutation>(distribution(generator));
}

int getRandomDigit() {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, N);
    return distrib(gen);
}

class Table {
private:
    vector<vector<int>> grid;
    int fitness = 0;

public:
    Table() : grid(N, vector<int>(N, 0)) {
        this->setFitness();
    }
    explicit Table(vector<vector<int>> g) : grid(std::move(g)) {
        this->setFitness();
    }

    vector<vector<int>> getGrid() {
        return this->grid;
    }

    vector<int> getPossible(int row, int col) {
        set<int> possible_digits = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        for (int j = 0; j < N; ++j) {
            if (this->grid[row][j] != 0) possible_digits.erase(this->grid[row][j]);
        }
        for (int i = 0; i < N; ++i) {
            if (this->grid[i][col] != 0) possible_digits.erase(this->grid[i][col]);
        }
        int square_row = (row / 3) * 3;
        int square_col = (col / 3) * 3;

        for (int i = square_row; i < square_row + 3; ++i) {
            for (int j = square_col; j < square_col + 3; ++j) {
                if (this->grid[i][j] != 0) possible_digits.erase(this->grid[i][j]);
            }
        }
        vector<int> result;
        for (int d : possible_digits) {
            result.push_back(d);
        }
        return result;
    }

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
//                cout << "No more obvious squares\n";
                break;
            }
        }
    }

    Table get_random_solution() {
        vector<vector<int>> solution = this->grid;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (grid[i][j] == 0) {
                    vector<int> possible_digits = this->getPossible(i, j);
                    if (!possible_digits.empty()) solution[i][j] = possible_digits[getRandom(0, int(possible_digits.size()) - 1)];
                    else solution[i][j] = getRandomDigit();
                }
            }
        }
        return Table(solution);
    }

    friend ostream &operator<<(ostream &out, const Table &t) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                out << t.grid[i][j] << " ";
            }
            out << endl;
        }
        return out;
    }

    friend istream &operator>>(istream &in, Table &t) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                char c;
                in >> c;
                if (c != '-' && c != '0') {
                    t.grid[i][j] = c - '0';
                } else t.grid[i][j] = 0;
            }
        }
        return in;
    }

    vector<set<int>> rowDigits() {
        vector<set<int>> res(9);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                res[i].insert(this->grid[i][j]);
            }
        }
        return res;
    }

    vector<set<int>> columnDigits() {
        vector<set<int>> res(9);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                res[j].insert(this->grid[i][j]);
            }
        }
        return res;
    }

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

    int getFitness() const {
        return this->fitness;
    }

    bool is_solved() const {
        return this->fitness == 243;
    }
};

class GeneticAlgorithm {
private:
    int population_size = 500;
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
    explicit GeneticAlgorithm(Table t) {
        t.predeterminedCells();
        this->table = t;
        this->table.setFitness();
    }

    Table getTable() {
        return this->table;
    }

    vector<Table> get_population() {
        vector<Table> population;
        for (int i = 0; i < this->population_size; ++i) {
            population.push_back(this->table.get_random_solution());
        }
        return population;
    }

    Table mutate(Table& t) const {
        vector<vector<int>> mutated_solution = t.getGrid();
        for (int i = 0; i < N; ++i) {
            if (getRandom(0.0, 1.0) < this->mutation_rate) {
                Mutation mutation_type = getRandomMutation();
                switch (mutation_type) {
                    case ROW:
                        swap(mutated_solution[i][getRandom(0, N-1)], mutated_solution[i][getRandom(0, N-1)]);
                        break;
                    case COLUMN:
                        swap(mutated_solution[getRandom(0, N-1)][i], mutated_solution[getRandom(0, N-1)][i]);
                        break;
                    case SQUARE:
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
                        swap(mutated_solution[i1][j1], mutated_solution[i2][j2]);
                        break;
                }
            }
        }
        return Table(mutated_solution);
    }

    static Table crossover(Table parent1, Table parent2) {
        int crossover_point = getRandomDigit();
        vector<vector<int>> child;
        for (int i = 0; i < crossover_point; ++i) {
            child.push_back(parent1.getGrid()[i]);
        }
        for (int i = crossover_point; i < N; ++i) {
            child.push_back(parent2.getGrid()[i]);
        }
        return Table(child);
    }

    static Table* get_solution(vector<Table> population) {
        int n = int(population.size());
        for (int i = 0; i < n; ++i) {
            if (population[i].is_solved()) return &population[i];
        }
        return nullptr;
    }

    static Table get_best_solution(vector<Table>& population) {
        int largest_fitness = MINUS_INFINITY;
        Table best_solution;
        for (const auto& solution : population) {
            if (solution.getFitness() > largest_fitness) {
                largest_fitness = solution.getFitness();
                best_solution = solution;
            }
        }
        return best_solution;
    }

    Table choose_parent(vector<Table> population) const {
        vector<Table> random_sample;
        int tables_number = int(population.size()) - 1;
        for (int i = 0; i < this->random_sample_size; ++i) {
            random_sample.push_back(population[getRandom(0, tables_number)]);
        }
        return get_best_solution(random_sample);
    }

    vector<Table> genetic_algorithm(vector<Table> population) {
        vector<Table> new_population;
        int num_elitism = int(int(population.size()) * this->elitism_portion);
        vector<Table> sorted_population = population;
        sort(sorted_population.begin(), sorted_population.end(), [](Table& a, Table& b) { return a.getFitness() > b.getFitness(); });
        for (int i = 0; i < num_elitism; ++i) {
            new_population.push_back(sorted_population[i]);
        }

        int num_random_selection = int((int(population.size()) - num_elitism) * this->random_selection_portion);
        int num_tournament_selection = int(population.size()) - num_elitism - num_random_selection;

        for (int i = 0; i < num_random_selection; ++i) {
            new_population.push_back(population[i]);
        }

        for (int i = 0; i < num_tournament_selection; ++i) {
            Table parent1 = this->choose_parent(population);
            Table parent2 = this->choose_parent(population);
            Table child = GeneticAlgorithm::crossover(parent1, parent2);
            new_population.push_back(child);
        }

        int new_population_size = int(new_population.size());
        for (int i = 0; i < new_population_size; ++i) {
            new_population[i] = this->mutate(new_population[i]);
        }

        auto cur_best_solution = GeneticAlgorithm::get_best_solution(new_population);
        int cur_best_fitness = cur_best_solution.getFitness();

        if (cur_best_fitness > this->best_fitness) {
            this->best_fitness = cur_best_fitness;
            this->stagnant_generations = 0;
            this->mutation_rate *= (1/this->mutation_rate_factor);
            this->mutation_rate = max(this->mutation_rate, this->min_mutation_rate);
        } else this->stagnant_generations += 1;

        if (this->stagnant_generations > this->stagnation_threshold) {
            this->mutation_rate *= this->mutation_rate_factor;
            this->mutation_rate = min(this->mutation_rate, this->max_mutation_rate);
            this->stagnant_generations = 0;
        }
        if (cur_best_solution.is_solved()) {
            vector<Table> answer = { cur_best_solution };
            return answer;
        }
        return new_population;
    }
};

void solve_sudoku(Table t) {
    GeneticAlgorithm sudoku_solver(std::move(t));

    if (sudoku_solver.getTable().getFitness() == 243) {
        cout << sudoku_solver.getTable();
        return;
    }
//    cout << "Solving" << endl;
    vector<Table> population = sudoku_solver.get_population();

    for (int generation = 0; generation < MAX_GENERATION; ++generation) {
        Table best_solution = GeneticAlgorithm::get_best_solution(population);
        int best_fitness = best_solution.getFitness();
        cout << "Generation" << generation + 1 << ": Best Fitness: " << best_fitness << "/243" << endl;

        population = sudoku_solver.genetic_algorithm(population);

        auto solution = GeneticAlgorithm::get_solution(population);
        if (solution != nullptr) {
            cout << *solution;
            return;
        }
    }
}

int main() {
    auto start = chrono::high_resolution_clock::now();
    Table s;
    ifstream inputFile("C:/didi/Inno/2 year 1 semester/introToAI/sudoku_ga/input.txt");

    if (inputFile.is_open()) {
        inputFile >> s;
        inputFile.close();
        cout << s << endl;
    } else {
//        cout << "Error opening file!" << endl;
        return 1;
    }
    solve_sudoku(s);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Execution time: " << elapsed.count() << " seconds" << endl;

    return 0;
}
