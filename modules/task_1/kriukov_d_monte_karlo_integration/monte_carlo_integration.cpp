// Copyright 2019 Kriukov Dmitry
#include <mpi.h>
#include <random>
#include <ctime>
#include <algorithm>
#include "../../../modules/task_1/kriukov_d_monte_karlo_integration/monte_carlo_integration.h"


double monteCarloIntegration(double lower_limit, double upper_limit, double(*pfunc)(double), int point_count) {
    if (point_count < 0)
        throw(1);

    std::mt19937 gen;
    std::uniform_real_distribution<> urd(0, 1);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int share = point_count / size;

    if (rank == 0) {
        gen.seed(static_cast<unsigned int>(time(0)));
        for (int proc = 1; proc < size - 1; proc++) {
            MPI_Send(pfunc, share, MPI_INT, proc, 0, MPI_COMM_WORLD);
        }
        if (size > 1)
            MPI_Send(pfunc, point_count - share * (size - 1), MPI_INT, size - 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Status status;
        MPI_Recv(pfunc, share, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

    double local_sum = 0.0;
    double global_sum = 0.0;

    for (int i = 0; i < share; i++) {
        local_sum += pfunc(lower_limit + urd(gen)*(upper_limit - lower_limit));
    }

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return global_sum *(upper_limit - lower_limit) / point_count;
}

