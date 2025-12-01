// nbody_cuda.cu
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <cstdlib>

namespace fs = std::filesystem;

const float Gf = 1.0f;
const float DTf = 0.01f;
const float SOFTENINGf = 1e-4f;
const float BOX_SIZEf = 1.0f;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(err) << "\n";           \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

__global__
void nbody_step_kernel(float *x, float *y,
                       float *vx, float *vy,
                       const float *m,
                       int n, float dt,
                       float G, float softening,
                       float box_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = x[i];
    float yi = y[i];
    float mi = m[i];

    float fxi = 0.0f;
    float fyi = 0.0f;

    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        float dx = x[j] - xi;
        float dy = y[j] - yi;
        float dist2 = dx * dx + dy * dy + softening;
        float invDist = rsqrtf(dist2);
        float invDist3 = invDist * invDist * invDist;
        float f = G * mi * m[j] * invDist3;
        fxi += f * dx;
        fyi += f * dy;
    }

    float ax = fxi / mi;
    float ay = fyi / mi;

    float vxi = vx[i] + ax * dt;
    float vyi = vy[i] + ay * dt;
    float xi_new = xi + vxi * dt;
    float yi_new = yi + vyi * dt;

    if (xi_new < 0.0f) {
        xi_new = 0.0f;
        vxi *= -1.0f;
    } else if (xi_new > box_size) {
        xi_new = box_size;
        vxi *= -1.0f;
    }

    if (yi_new < 0.0f) {
        yi_new = 0.0f;
        vyi *= -1.0f;
    } else if (yi_new > box_size) {
        yi_new = box_size;
        vyi *= -1.0f;
    }

    x[i] = xi_new;
    y[i] = yi_new;
    vx[i] = vxi;
    vy[i] = vyi;
}

void init_bodies_host(float *x, float *y,
                      float *vx, float *vy,
                      float *m, int n,
                      unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> pos_dist(0.0f, BOX_SIZEf);
    std::uniform_real_distribution<float> vel_dist(-0.01f, 0.01f);
    std::uniform_real_distribution<float> mass_dist(0.5f, 1.5f);

    for (int i = 0; i < n; ++i) {
        x[i] = pos_dist(gen);
        y[i] = pos_dist(gen);
        vx[i] = vel_dist(gen);
        vy[i] = vel_dist(gen);
        m[i] = mass_dist(gen);
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " N num_steps output_dir\n";
        return 1;
    }

    int N = std::stoi(argv[1]);
    int num_steps = std::stoi(argv[2]);
    std::string output_dir = argv[3];

    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    float *h_x = nullptr;
    float *h_y = nullptr;
    float *h_vx = nullptr;
    float *h_vy = nullptr;
    float *h_m = nullptr;

    h_x = static_cast<float*>(std::malloc(bytes));
    h_y = static_cast<float*>(std::malloc(bytes));
    h_vx = static_cast<float*>(std::malloc(bytes));
    h_vy = static_cast<float*>(std::malloc(bytes));
    h_m  = static_cast<float*>(std::malloc(bytes));

    if (!h_x || !h_y || !h_vx || !h_vy || !h_m) {
        std::cerr << "Host allocation failed\n";
        return 1;
    }

    init_bodies_host(h_x, h_y, h_vx, h_vy, h_m, N);

    float *d_x = nullptr;
    float *d_y = nullptr;
    float *d_vx = nullptr;
    float *d_vy = nullptr;
    float *d_m = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_vx, bytes));
    CUDA_CHECK(cudaMalloc(&d_vy, bytes));
    CUDA_CHECK(cudaMalloc(&d_m, bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m, h_m, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < num_steps; ++step) {
        nbody_step_kernel<<<gridSize, blockSize>>>(
            d_x, d_y, d_vx, d_vy, d_m,
            N, DTf, Gf, SOFTENINGf, BOX_SIZEf
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "CUDA time: " << elapsed << " s\n";

    CUDA_CHECK(cudaMemcpy(h_x, d_x, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vx, d_vx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vy, d_vy, bytes, cudaMemcpyDeviceToHost));

    fs::path out_dir(output_dir);
    try {
        fs::create_directories(out_dir);
    } catch (const std::exception &e) {
        std::cerr << "Failed to create output directory: " << e.what() << "\n";
        return 1;
    }

    fs::path out_path =
        out_dir / ("cuda_N" + std::to_string(N) +
                   "_steps" + std::to_string(num_steps) + ".txt");

    std::ofstream ofs(out_path);
    if (!ofs) {
        std::cerr << "Error opening output file: " << out_path << "\n";
        return 1;
    }

    for (int i = 0; i < N; ++i) {
        ofs << i << " "
            << h_x[i] << " " << h_y[i] << " "
            << h_vx[i] << " " << h_vy[i] << "\n";
    }
    ofs.close();

    std::cout << "CUDA output written to: " << out_path << "\n";

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_vx));
    CUDA_CHECK(cudaFree(d_vy));
    CUDA_CHECK(cudaFree(d_m));

    std::free(h_x);
    std::free(h_y);
    std::free(h_vx);
    std::free(h_vy);
    std::free(h_m);

    return 0;
}
