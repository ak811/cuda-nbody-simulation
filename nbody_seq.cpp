// nbody_seq.cpp
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Body {
    double x, y;
    double vx, vy;
    double m;
};

const double G = 1.0;
const double DT = 0.01;
const double SOFTENING = 1e-4;
const double BOX_SIZE = 1.0;

void init_bodies(std::vector<Body> &bodies, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> pos_dist(0.0, BOX_SIZE);
    std::uniform_real_distribution<double> vel_dist(-0.01, 0.01);
    std::uniform_real_distribution<double> mass_dist(0.5, 1.5);

    for (auto &b : bodies) {
        b.x = pos_dist(gen);
        b.y = pos_dist(gen);
        b.vx = vel_dist(gen);
        b.vy = vel_dist(gen);
        b.m = mass_dist(gen);
    }
}

void step_sequential(std::vector<Body> &bodies) {
    int n = static_cast<int>(bodies.size());
    std::vector<double> fx(n, 0.0), fy(n, 0.0);

    // Compute forces
    for (int i = 0; i < n; ++i) {
        double xi = bodies[i].x;
        double yi = bodies[i].y;
        double mi = bodies[i].m;
        double fxi = 0.0;
        double fyi = 0.0;

        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            double dx = bodies[j].x - xi;
            double dy = bodies[j].y - yi;
            double dist2 = dx * dx + dy * dy + SOFTENING;
            double invDist = 1.0 / std::sqrt(dist2);
            double invDist3 = invDist * invDist * invDist;
            double f = G * mi * bodies[j].m * invDist3;
            fxi += f * dx;
            fyi += f * dy;
        }
        fx[i] = fxi;
        fy[i] = fyi;
    }

    // Update velocities and positions, apply boundary reflections
    for (int i = 0; i < n; ++i) {
        double ax = fx[i] / bodies[i].m;
        double ay = fy[i] / bodies[i].m;

        bodies[i].vx += ax * DT;
        bodies[i].vy += ay * DT;
        bodies[i].x += bodies[i].vx * DT;
        bodies[i].y += bodies[i].vy * DT;

        if (bodies[i].x < 0.0) {
            bodies[i].x = 0.0;
            bodies[i].vx *= -1.0;
        } else if (bodies[i].x > BOX_SIZE) {
            bodies[i].x = BOX_SIZE;
            bodies[i].vx *= -1.0;
        }

        if (bodies[i].y < 0.0) {
            bodies[i].y = 0.0;
            bodies[i].vy *= -1.0;
        } else if (bodies[i].y > BOX_SIZE) {
            bodies[i].y = BOX_SIZE;
            bodies[i].vy *= -1.0;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " N num_steps output_dir write_trajectories(0 or 1)\n";
        return 1;
    }

    int N = std::stoi(argv[1]);
    int num_steps = std::stoi(argv[2]);
    std::string output_dir = argv[3];
    int write_traj = std::stoi(argv[4]);

    std::vector<Body> bodies(N);
    init_bodies(bodies);

    fs::path out_dir(output_dir);
    try {
        fs::create_directories(out_dir);
    } catch (const std::exception &e) {
        std::cerr << "Failed to create output directory: " << e.what() << "\n";
        return 1;
    }

    std::string suffix = write_traj ? "traj" : "final";
    fs::path out_path =
        out_dir / ("seq_N" + std::to_string(N) +
                   "_steps" + std::to_string(num_steps) +
                   "_" + suffix + ".txt");

    std::ofstream ofs(out_path);
    if (!ofs) {
        std::cerr << "Error opening output file: " << out_path << "\n";
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < num_steps; ++step) {
        step_sequential(bodies);

        if (write_traj) {
            for (int i = 0; i < N; ++i) {
                ofs << step << " " << i << " "
                    << bodies[i].x << " " << bodies[i].y << " "
                    << bodies[i].vx << " " << bodies[i].vy << "\n";
            }
        }
    }

    if (!write_traj) {
        for (int i = 0; i < N; ++i) {
            ofs << i << " "
                << bodies[i].x << " " << bodies[i].y << " "
                << bodies[i].vx << " " << bodies[i].vy << "\n";
        }
    }

    ofs.close();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "Sequential time: " << elapsed << " s\n";
    std::cout << "Sequential output written to: " << out_path << "\n";

    return 0;
}
