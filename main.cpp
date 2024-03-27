#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <immintrin.h>

void initializeMatrix(double* matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = static_cast<double>(rand()) / RAND_MAX; // случайное число от 0 до 1
    }
}

// Последовательное умножение матриц
void DGEMM(double* A, double* B, double* C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double sum = 0.0;
            for (int k = 0; k < size; ++k) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

// Оптимизированная функция с построчным доступом к памяти
void DGEMM_opt_1(double* A, double* B, double* C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < size; ++k) {
            for (int j = 0; j < size; ++j) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

// Оптимизированная функция с блочным доступом к памяти
void DGEMM_opt_2(double* A, double* B, double* C, int size, int block_size) {
    for (int i0 = 0; i0 < size; i0 += block_size) {
        for (int j0 = 0; j0 < size; j0 += block_size) {
            for (int k0 = 0; k0 < size; k0 += block_size) {
                for (int i = i0; i < std::min(i0 + block_size, size); ++i) {
                    for (int j = j0; j < std::min(j0 + block_size, size); ++j) {
                        for (int k = k0; k < std::min(k0 + block_size, size); ++k) {
                            C[i * size + j] += A[i * size + k] * B[k * size + j];
                        }
                    }
                }
            }
        }
    }
}

// Оптимизированная функция с векторизацией кода
void DGEMM_opt_3(double* A, double* B, double* C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            __m256d sum = _mm256_set1_pd(0.0);
            for (int k = 0; k < size; k += 4) {
                __m256d a_vec = _mm256_loadu_pd(&A[i * size + k]);
                __m256d b_vec = _mm256_loadu_pd(&B[k * size + j]);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(a_vec, b_vec));
            }
            double result[4];
            _mm256_storeu_pd(result, sum);
            C[i * size + j] = result[0] + result[1] + result[2] + result[3];
        }
    }
}
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " method_index matrix_size [block_size]" << std::endl;
        std::cerr << "method_index: 0 - DGEMM, 1 - DGEMM_opt_1, 2 - DGEMM_opt_2, 3 - DGEMM_opt_3" << std::endl;
        return 1;
    }

    int method_index = std::atoi(argv[1]);
    int size = std::atoi(argv[2]);
    int block_size = (argc > 3) ? std::atoi(argv[3]) : 16;

    double* A = new double[size * size];
    double* B = new double[size * size];
    double* C = new double[size * size];

    srand(time(NULL));
    initializeMatrix(A, size);
    initializeMatrix(B, size);

    auto start = std::chrono::high_resolution_clock::now();
    if (method_index == 0) {
        DGEMM(A, B, C, size);
    } else if (method_index == 1) {
        DGEMM_opt_1(A, B, C, size);
    } else if (method_index == 2) {
        DGEMM_opt_2(A, B, C, size, block_size);
    } else if (method_index == 3) {
        DGEMM_opt_3(A, B, C, size);
    } else {
        std::cerr << "Invalid method_index" << std::endl;
        return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time for method " << method_index << ": " << elapsed_seconds.count() << "s\n";

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

/*
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " matrix_size [block_size]" << std::endl;
        return 1;
    }

    int size = std::atoi(argv[1]);
    int block_size = (argc > 2) ? std::atoi(argv[2]) : 1000;

    double* A = new double[size * size];
    double* B = new double[size * size];
    double* C = new double[size * size];

    srand(time(NULL));
    initializeMatrix(A, size);
    initializeMatrix(B, size);

    auto start = std::chrono::high_resolution_clock::now();
    DGEMM(A, B, C, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time (DGEMM): " << elapsed_seconds.count() << "s\n";

    start = std::chrono::high_resolution_clock::now();
    DGEMM_opt_1(A, B, C, size);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Elapsed time (DGEMM_opt_1): " << elapsed_seconds.count() << "s\n";

    start = std::chrono::high_resolution_clock::now();
    DGEMM_opt_2(A, B, C, size, block_size);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Elapsed time (DGEMM_opt_2 with block size " << block_size << "): " << elapsed_seconds.count() << "s\n";

    start = std::chrono::high_resolution_clock::now();
    DGEMM_opt_3(A, B, C, size);
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Elapsed time (DGEMM_opt_3): " << elapsed_seconds.count() << "s\n";

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
} */
/*
int main() {
    const int size = 1000; // Устанавливаем размер матрицы
    const int block_sizes[] = {2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000};
    const int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);

    double* A = new double[size * size];
    double* B = new double[size * size];
    double* C = new double[size * size];

    srand(time(NULL));
    initializeMatrix(A, size);
    initializeMatrix(B, size);

    for (int i = 0; i < num_block_sizes; ++i) {
        int block_size = block_sizes[i];
        auto start = std::chrono::high_resolution_clock::now();
        DGEMM_opt_2(A, B, C, size, block_size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Elapsed time (DGEMM_opt_2 with block size " << block_size << "): " << elapsed_seconds.count() << "s\n";
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
} */
/*
int main() {
    const int sizes[] = {1000, 2000, 3000, 4000, 5000, 6000};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int block_size = 1000;

    for (int i = 0; i < num_sizes; ++i) {
        int size = sizes[i];

        double* A = new double[size * size];
        double* B = new double[size * size];
        double* C = new double[size * size];

        srand(time(NULL));
        initializeMatrix(A, size);
        initializeMatrix(B, size);

        auto start = std::chrono::high_resolution_clock::now();
        DGEMM(A, B, C, size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Elapsed time (DGEMM) for size " << size << ": " << elapsed_seconds.count() << "s\n";

        start = std::chrono::high_resolution_clock::now();
        DGEMM_opt_1(A, B, C, size);
        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "Elapsed time (DGEMM_opt_1) for size " << size << ": " << elapsed_seconds.count() << "s\n";

        start = std::chrono::high_resolution_clock::now();
        DGEMM_opt_2(A, B, C, size, block_size);
        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "Elapsed time (DGEMM_opt_2 with block size " << block_size << ") for size " << size << ": " << elapsed_seconds.count() << "s\n";

        start = std::chrono::high_resolution_clock::now();
        DGEMM_opt_3(A, B, C, size);
        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        std::cout << "Elapsed time (DGEMM_opt_3) for size " << size << ": " << elapsed_seconds.count() << "s\n";

        delete[] A;
        delete[] B;
        delete[] C;
    }

    return 0;
} */