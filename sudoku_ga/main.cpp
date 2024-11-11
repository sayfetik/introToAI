#include <iostream>
#include <utility>
#include <vector>
#include <random>
#include <set>
#include <algorithm>
#include <fstream>

using namespace std;

const int N = 9;
const int MINUS_INFINITY = numeric_limits<int>::min();


int getRandomNumber(int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(1, n);
    return distrib(gen);
}

class Table {
private:
    vector<vector<int>> grid;
    int fitness = 0;
    set<pair<int, int>> filled;

public:
    Table() : grid(N, vector<int>(N, 0)) {}
    explicit Table(vector<vector<int>> g) : grid(std::move(g)) {}

    bool operator>(const Table &other) const {
        return fitness >= other.fitness;
    }

    bool operator<(const Table &other) const {
        return fitness < other.fitness;
    }

    void addFilled(int i, int j) {
        this->filled.insert(make_pair(i, j));
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
        vector<vector<int>> solution(N, vector<int>(N, 0));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (grid[i][j] != 0) solution[i][j] = this->grid[i][j];
                else {
                    vector<int> possible_digits = this->getPossible(i, j);
                    if (!possible_digits.empty()) solution[i][j] = possible_digits[getRandomNumber(int(possible_digits.size()))];
                    else solution[i][j] = getRandomNumber(N);
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
                    t.addFilled(i, j);
                } else t.grid[i][j] = 0;
            }
        }
        return in;
    }

    bool isEmpty(int i, int j) {
        return grid[i][j] == -1;
    }

    [[nodiscard]] int getFitness() const {
        return this->fitness;
    }

    int getCell(int i, int j) {
        return this->grid[i][j];
    }

    void setCell(int i, int j, int digit) {
        this->grid[i][j] = digit;
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

    int setFitness() {
        int res = 0;
        vector<set<int>> rowDigits = this->rowDigits();
        vector<set<int>> columnDigits = this->columnDigits();
        vector<set<int>> squareDigits = this->squareDigits();
        for (int i = 0; i < N; ++i) {
            res += int(rowDigits[i].size()) + int(columnDigits[i].size()) + int(squareDigits[i].size());
        }
        this->fitness = res;
        return res;
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

    vector<Table> get_initial_solution() {
        vector<Table> population;
        for (int i = 0; i < this->population_size; ++i) {
            population.push_back(this->table.get_random_solution());
        }
        return population;
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
