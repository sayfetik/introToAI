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


int getRandomNumber1(int n) {
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

    vector<vector<int>> getGrid() {
        return this->grid;
    };

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
        int square_row = (row / 3) * 3; // maybe mistake (целочисленное деление должно быть)
        int square_col = (col / 3) * 3; // maybe mistake (целочисленное деление должно быть)

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
                    if (!possible_digits.empty()) solution[i][j] = possible_digits[getRandomNumber1(possible_digits.size())];
                    else solution[i][j] = getRandomNumber1(N);
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

    int getFitness() const {
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
            res += rowDigits[i].size() + columnDigits[i].size() + squareDigits[i].size();
        }
        this->fitness = res;
        return res;
    }
};

vector<Table> create_initial_population(Table initial_grid) {
    vector<Table> population(100);
    for (int k = 0; k < 100; ++k) {
        Table t = initial_grid;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (t.isEmpty(i, j)) t.setCell(i, j, getRandomNumber1(N));
            }
        }
        t.setFitness();
        population[k] = t;
    }
    sort(population.begin(), population.end(), [](const Table &a, const Table &b) {
        return a.getFitness() > b.getFitness();
    });
    return population;
}

Table crossover(Table &parent1, Table &parent2) {
    Table child;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (rand() % 2 == 0) {
                child.setCell(i, j, parent1.getCell(i, j));
            } else {
                child.setCell(i, j, parent2.getCell(i, j));
            }
        }
    }
    child.setFitness();
    return child;
}

Table mutation(Table t) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, N - 1);
    uniform_int_distribution<> numberDistrib(1, 9);

    // Увеличенная вероятность мутации
    float mutationRate = 0.15;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (distrib(gen) < mutationRate * N && t.isEmpty(i, j)) {
                int newValue = numberDistrib(gen);
                // Избегаем установки значения, которое уже есть в строке, столбце или квадрате
                set<int> row = t.rowDigits()[i];
                set<int> col = t.columnDigits()[j];
                int squareIndex = (i / 3) * 3 + (j / 3);
                set<int> square = t.squareDigits()[squareIndex];

                while (row.count(newValue) || col.count(newValue) || square.count(newValue)) {
                    newValue = numberDistrib(gen);
                }
                t.setCell(i, j, newValue);
            }
        }
    }

    t.setFitness();
    return t;
}

Table solve_sudoku(Table initial_grid) {
    // Создаем начальное население
    vector<Table> population = create_initial_population(initial_grid);

    // Параметры для эволюции
    float crossoverRate = 0.7;  // 70% вероятность для кроссовера
    float mutationRate = 0.3;  // 30% вероятность для мутации
    int generation = 0;
    int stagnationCount = 0; // Счетчик стагнации

    while (true) {
        // Сортируем популяцию по фитнесу
        sort(population.begin(), population.end(), [](const Table& a, const Table& b) {
            return a.getFitness() > b.getFitness();
        });

        // Вывод текущего поколения и лучшего фитнеса
        cout << "Generation: " << generation << ", Best Fitness: " << population[0].getFitness() << endl;

        // Если решение найдено (например, фитнес = 81)
        if (population[0].getFitness() == 80.19) {
            cout << "Solution found at generation " << generation << "!" << endl;
            cout << population[0] << endl;
            return population[0];
        }

        // Если лучшая особь не улучшилась
        static int bestFitness = population[0].getFitness();
        if (population[0].getFitness() == bestFitness) {
            stagnationCount++;
        } else {
            stagnationCount = 0; // Сброс, если произошло улучшение
            bestFitness = population[0].getFitness();
        }

        // Внедрение "перезапуска" при застое
        if (stagnationCount > 100) {  // Если 100 поколений без улучшения
            cout << "Stagnation detected. Introducing random individuals..." << endl;
            for (int i = population.size() / 2; i < population.size(); ++i) {
//                population[i] = create_random_individual();  // Функция для создания случайных особей
            }
            stagnationCount = 0;  // Сброс счетчика
        }

        // Создаем новое поколение
        vector<Table> newPopulation;

        // Применяем кроссовер и мутацию для создания нового поколения
        for (int i = 0; i < population.size() / 2; ++i) {
            Table parent1 = population[i];
            Table parent2 = population[i + 1];

            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> distrib(0.0, 1.0);
            if (distrib(gen) < crossoverRate) {
                Table child = crossover(parent1, parent2);
                newPopulation.push_back(child);
            } else {
                newPopulation.push_back(parent1);
            }

            if (distrib(gen) < mutationRate) {
                newPopulation.back() = mutation(newPopulation.back());
            }
        }

        // Обновляем популяцию
        population = newPopulation;
        generation++;
    }
}

int main1() {
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

//    Table solution = solve_sudoku(s);  // Решение Судоку с использованием генетического алгоритма
//
//    cout << "Final Solution: " << endl;
//    cout << solution << endl;

    return 0;
}
