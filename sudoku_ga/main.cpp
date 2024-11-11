#include <iostream>
#include <utility>
#include <vector>
#include <random>
#include <set>
#include <algorithm>
#include <fstream>
#include <type_traits>

using namespace std;

const int N = 9;
const int MINUS_INFINITY = numeric_limits<int>::min();

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

int getRandomDigit(int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, n);
    return distrib(gen);
}

class Table {
private:
    vector<vector<int>> grid;
    int fitness = 0;

public:
    Table() : grid(N, vector<int>(N, 0)) {}
    explicit Table(vector<vector<int>> g) : grid(std::move(g)) {}

    bool operator>(const Table &other) const {
        return fitness >= other.fitness;
    }

    bool operator<(const Table &other) const {
        return fitness < other.fitness;
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
            bool squares_filled = false;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    if (this->grid[i][j] == 0) {
                        vector<int> possible_digits = this->getPossible(i, j);
                        if (possible_digits.size() == 1) {
                            this->grid[i][j] = possible_digits[0];
                            squares_filled = true;
                        }
                    }
                }
            }
            if (!squares_filled) {
                cout << "No more obvious squares\n";
                break;
            }
        }
        cout << *this;
    }

    Table get_random_solution() {
        vector<vector<int>> solution = this->grid;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (grid[i][j] == 0) {
                    vector<int> possible_digits = this->getPossible(i, j);
                    if (!possible_digits.empty()) solution[i][j] = possible_digits[getRandomDigit(int(possible_digits.size()))];
                    else solution[i][j] = getRandomDigit(N);
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
                if (c != '-') {
                    t.grid[i][j] = c - '0';
                } else t.grid[i][j] = 0;
            }
        }
        return in;
    }

    void setCell(int i, int j, int digit) {
        this->grid[i][j] = digit;
    }
};

class GeneticAlgorithm {
private:
    int population_size = 1000;
    int max_generations = 100000;
    double mutation_rate = 0.04;
    double mutation_rate_factor = 1.5;
    int stagnation_threshold = 50;
    int tournament_size = 10;
    double random_selection_portion = 0.1;
    double elitism_portion = 0.1;
    int generations_without_improvement = 0;
    int best_fitness_so_far = MINUS_INFINITY;
    int best_fitness = -1;
    int stagnant_generations = 0;
    double max_mutation_rate = 0.05;
    double min_mutation_rate = 0.01;
    Table table;
public:
    explicit GeneticAlgorithm(Table t) : table(std::move(t)) {}

    vector<Table> get_population() {
        vector<Table> population;
        for (int i = 0; i < this->population_size; ++i) {
            population.push_back(this->table.get_random_solution());
        }
        return population;
    }

    Table mutate() {
        vector<vector<int>> mutated_solution = this->table.getGrid();
        for (int i = 0; i < N; ++i) {
            if (getRandom(0.0, 1.0) < this->mutation_rate) {
                Mutation mutation_type = getRandomMutation();
                switch (mutation_type) {
                    case ROW:
                        swap(mutated_solution[i][getRandomDigit(N)], mutated_solution[i][getRandomDigit(N)]);
                        break;
                    case COLUMN:
                        swap(mutated_solution[getRandomDigit(N)][i], mutated_solution[getRandomDigit(N)][i]);
                        break;
                    case SQUARE:
                        int square_row = (i / 3) * 3;
                        int square_col = (i % 3) * 3;
                        vector<pair<int, int>> square_cells;
                        for (int j = square_row; j < square_row + 3; ++j) {
                            for (int k = square_col; k < square_col + 3; ++k) {
                                square_cells.emplace_back(j, k);
                            }
                        }
                        int size_square_cells = int(square_cells.size());
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
};

int main() {
    Table s;
    ifstream inputFile("C:/Users/Public/CLionProjects/AI/input.txt");

    if (inputFile.is_open()) {
        inputFile >> s;
        inputFile.close();
        cout << s << endl;
        s.predeterminedCells();
    } else {
        cout << "Error opening file!" << endl;
        return 1;
    }

    return 0;
}
