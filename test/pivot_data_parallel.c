#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<mpi.h>
#include<omp.h>
int PivotCompare(int k, int* pivot, int* last_pivot) {
    int i;
    for(i=0; i<k && pivot[i]==last_pivot[i]; ++i){}
    return i;
}

// Calculate sum of distance while combining different pivots. Complexity : O( n^2 )
double SumDistance(const int k, const int n, const int dim, const int size, 
                    const int rank, double* coord, int* pivots, int* last_pivots, double* rebuiltCoord_local){
    int n_local = (n+size-1) / size;
    int ni;
    int n_end = (rank==size-1)? n-(size-1)*n_local : n_local;
    int k_start;

    // Compare this pivots and last pivots
    if(last_pivots[0] == -1) k_start = 0;
    else k_start = PivotCompare(k, pivots, last_pivots);

    // Rebuild coordinates. New coordinate of one point is its distance to each pivot.
    #pragma omp parallel for
    for(ni=0; ni<n_end; ++ni) {
        int ki;
        int ni_global = rank * n_local + ni;
        for(ki=k_start; ki<k; ++ki) {
            double distance = 0;
            int pivoti = pivots[ki];
            int j;
            for(j=0; j<dim; ++j) {
                distance += pow(coord[pivoti*dim + j]-coord[ni_global*dim + j], 2);
            }
            rebuiltCoord_local[ni*k + ki] = sqrt(distance);
        }
    }

    double* rebuiltCoord = (double*)malloc(sizeof(double) * n_local * size * k);
    MPI_Allgather(rebuiltCoord_local, n_local*k, MPI_DOUBLE, rebuiltCoord, n_local*k, MPI_DOUBLE, MPI_COMM_WORLD);

    // Calculate the sum of Chebyshev distance with rebuilt coordinates between every points
    double chebyshevSum_local = 0, chebyshevSum = 0;
    #pragma omp parallel for reduction(+:chebyshevSum_local)
    for(ni=0; ni<n_end; ni++){
        int j;
        int ni_global = rank * n_local + ni;
        for(j=0; j<n; ++j){
            double chebyshev = 0;
            int ki;
            for(ki=0; ki<k; ki++){
                double dis = fabs(rebuiltCoord[ni_global*k + ki] - rebuiltCoord[j*k + ki]);
                chebyshev = dis>chebyshev ? dis : chebyshev;
            }
            chebyshevSum_local += chebyshev;
        }
    }
    MPI_Allreduce(&chebyshevSum_local, &chebyshevSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    free(rebuiltCoord);
    return chebyshevSum;
}

int nextCombination(int n, int k, int pivots[]) {
    int lastNotEqualOffset = k-1;
    while (pivots[lastNotEqualOffset] == n-k+(lastNotEqualOffset+1)) {
        lastNotEqualOffset--;
    }
    if (lastNotEqualOffset<0) {
        return 0;
    }
    pivots[lastNotEqualOffset]++;
    for (int i=lastNotEqualOffset+1; i<k; i++) {
        pivots[i] = pivots[lastNotEqualOffset]+(i-lastNotEqualOffset);
    }
    return 1;
}

// Recursive function Combination() : combine pivots and calculate the sum of distance while combining different pivots.
// ki  : current depth of the recursion
// k   : number of pivots
// n   : number of points
// dim : dimension of metric space
// M   : number of combinations to store
// coord  : coordinates of points
// pivots : indexes of pivots
// maxDistanceSum  : the largest M distance sum
// maxDisSumPivots : the top M pivots combinations
// minDistanceSum  : the smallest M distance sum
// minDisSumPivots : the bottom M pivots combinations
void Combination(const int k, const int n, const int dim, const int M, const int size, const int rank,
                double* coord, int* pivots, double* maxDistanceSum, int* maxDisSumPivots, double* minDistanceSum, int* minDisSumPivots){
        int n_local = (n+size-1) / size;
        int* last_pivots = (int*)malloc(sizeof(int) * k);
        double* last_coord_local = (double*)malloc(sizeof(double) * n_local * k);
        int i;
        for(i=0; i<k; ++i) {
            pivots[i] = i;
            last_pivots[i] = -1;
        }
        do{
            // Calculate sum of distance while combining different pivots.
            double distanceSum = SumDistance(k, n, dim, size, rank, coord, pivots, last_pivots, last_coord_local);
            int kj;

            // sort
            int a;

            switch(rank){
                case 0:
                maxDistanceSum[M] = distanceSum;
                for(kj=0; kj<k; kj++){
                    maxDisSumPivots[M*k + kj] = pivots[kj];
                }
                for(a=M; a>0; a--){
                    if(maxDistanceSum[a] > maxDistanceSum[a-1]){
                        double temp = maxDistanceSum[a];
                        maxDistanceSum[a] = maxDistanceSum[a-1];
                        maxDistanceSum[a-1] = temp;
                        for(kj=0; kj<k; kj++){
                            int temp = maxDisSumPivots[a*k + kj];
                            maxDisSumPivots[a*k + kj] = maxDisSumPivots[(a-1)*k + kj];
                            maxDisSumPivots[(a-1)*k + kj] = temp;
                        }
                    }
                }
                break;

                case 1:
                minDistanceSum[M] = distanceSum;
                for(kj=0; kj<k; kj++){
                    minDisSumPivots[M*k + kj] = pivots[kj];
                }
                for(a=M; a>0; a--){
                    if(minDistanceSum[a] < minDistanceSum[a-1]){
                        double temp = minDistanceSum[a];
                        minDistanceSum[a] = minDistanceSum[a-1];
                        minDistanceSum[a-1] = temp;
                        for(kj=0; kj<k; kj++){
                            int temp = minDisSumPivots[a*k + kj];
                            minDisSumPivots[a*k + kj] = minDisSumPivots[(a-1)*k + kj];
                            minDisSumPivots[(a-1)*k + kj] = temp;
                        }
                    }
                }
                break;
            }

            for(i=0; i<k; ++i) {
                last_pivots[i] = pivots[i];
            }

            /** Iteration Log : pivots computed, best pivots, max distance sum, min distance sum pivots, min distance sum
            *** You can delete the logging code. **/
            // if(rank==0)
            // {
            //     int kj;
                // for(kj=0; kj<k; kj++){
                //     printf("%d ", pivots[kj]);
                // }
                // printf("%lf\n", distanceSum);
            //     putchar('\t');
            //     for(kj=0; kj<k; kj++){
            //         printf("%d ", maxDisSumPivots[kj]);
            //     }
            //     printf("%lf\t", maxDistanceSum[0]);
            // }
            // else if(rank==1)
            // {
            //     for(kj=0; kj<k; kj++){
            //         printf("%d ", minDisSumPivots[kj]);
            //     }
            //     printf("%lf\n", minDistanceSum[0]);
            // }
        }while(nextCombination(n-1, k, pivots));

        free(last_pivots);
        free(last_coord_local);
    return;
}

int main(int argc, char* argv[]){
    MPI_Init(NULL, NULL);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(size < 2){
        puts("This should run by at least 2 nodes.");
        MPI_Finalize();
        exit(-1);
    }

    omp_set_num_threads(24);

    // M : number of combinations to store
    const int M = 1000;
    // dim : dimension of metric space
    int dim;
    // n : number of points
    int n;
    // k : number of pivots
    int k;
    // File pointer
    FILE* file;

    // Read with rank 0 only
    if(rank == 0)
    {
        // filename : input file namespace
        char* filename = (char*)"uniformvector-2dim-5h.txt";
        if( argc==2 ) {
            filename = argv[1];
        }  else if(argc != 1) {
            printf("Usage: ./pivot <filename>\n");
            return -1;
        }

        // Read parameter
        file = fopen(filename, "r");
        if( file == NULL ) {
            printf("%s file not found.\n", filename);
            return -1;
        }
        fscanf(file, "%d", &dim);
        fscanf(file, "%d", &n);
        fscanf(file, "%d", &k);
        printf("dim = %d, n = %d, k = %d\n", dim, n, k);
    }

    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    double* coord = (double*)malloc(sizeof(double) * dim * n);

    if(rank==0)
    {
        // Read Data
        int i;
        for(i=0; i<n; i++){
            int j;
            for(j=0; j<dim; j++){
                fscanf(file, "%lf", &coord[i*dim + j]);
            }
        }
        fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Start timing
    struct timeval start;
    gettimeofday(&start, NULL);

    // Broadcast the coordinate
    MPI_Bcast(coord, n*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int i;
    // maxDistanceSum : the largest M distance sum
    double* maxDistanceSum = (double*)malloc(sizeof(double) * (M+1));
    // maxDisSumPivots : the top M pivots combinations
    int* maxDisSumPivots = (int*)malloc(sizeof(int) * k * (M+1));
    // minDistanceSum : the smallest M distance sum
    double* minDistanceSum = (double*)malloc(sizeof(double) * (M+1));
    // minDisSumPivots : the bottom M pivots combinations
    int* minDisSumPivots = (int*)malloc(sizeof(int) * k * (M+1));

    int ki;
    switch(rank){
        case 0:
        for(i=0; i<M; i++){
            maxDistanceSum[i] = 0;
            for(ki=0; ki<k; ki++){
                maxDisSumPivots[i*k + ki] = 0;
            }
        }
        break;

        case 1:
        for(i=0; i<M; i++){
            minDistanceSum[i] = __DBL_MAX__;
            for(ki=0; ki<k; ki++){
                minDisSumPivots[i*k + ki] = 0;
            }
        }
        break;
    }

    // temp : indexes of pivots with dummy array head
    int* temp = (int*)malloc(sizeof(int) * k);

    // Main loop. Combine different pivots with recursive function and evaluate them. Complexity : O( n^(k+2) )
    Combination(k, n, dim, M, size, rank, coord, temp, maxDistanceSum, maxDisSumPivots, minDistanceSum, minDisSumPivots);
    MPI_Bcast(maxDistanceSum, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(minDistanceSum, M, MPI_DOUBLE, 1, MPI_COMM_WORLD);
    MPI_Bcast(maxDisSumPivots, M * k, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(minDisSumPivots, M * k, MPI_INT, 1, MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);
    
    // End timing
    struct timeval end;
    gettimeofday (&end, NULL);

    if(rank == 0)
    {
        printf("Using time : %f ms\n", (end.tv_sec-start.tv_sec)*1000.0+(end.tv_usec-start.tv_usec)/1000.0);

        // Store the result
        int i;
        char result_name[100];
        sprintf(result_name,"result1D%03dN%03dK%03d",dim,n,k);
        FILE* out = fopen(result_name, "w");
        for(i=0; i<M; i++){
            int ki;
            for(ki=0; ki<k-1; ki++){
                fprintf(out, "%d ", maxDisSumPivots[i*k + ki]);
            }
            fprintf(out, "%d\n", maxDisSumPivots[i*k + k-1]);
        }
        for(i=0; i<M; i++){
            int ki;
            for(ki=0; ki<k-1; ki++){
                fprintf(out, "%d ", minDisSumPivots[i*k + ki]);
            }
            fprintf(out, "%d\n", minDisSumPivots[i*k + k-1]);
        }
        fclose(out);
    }

        // Log
        switch(rank){
            case 0:
            printf("max : ");
            for(ki=0; ki<k; ki++){
                printf("%d ", maxDisSumPivots[ki]);
            }
            printf("%lf\n", maxDistanceSum[0]);
            break;

            case 1:
            printf("min : ");
            for(ki=0; ki<k; ki++){
                printf("%d ", minDisSumPivots[ki]);
            }
            printf("%lf\n", minDistanceSum[0]);
            break;

            // for(i=0; i<M; i++){
                // int ki;
                // for(ki=0; ki<k; ki++){
                    // printf("%d\t", maxDisSumPivots[i*k + ki]);
                // }
                // printf("%lf\n", maxDistanceSum[i]);
            // }
            // for(i=0; i<M; i++){
                // int ki;
                // for(ki=0; ki<k; ki++){
                    // printf("%d\t", minDisSumPivots[i*k + ki]);
                // }
                // printf("%lf\n", minDistanceSum[i]);
            // }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
