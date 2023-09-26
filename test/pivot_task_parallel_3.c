#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<mpi.h>
#include<omp.h>

#define MAX 0
#define MIN 1

// Compare with last pivots
int PivotCompare(int k, int* pivot, int* last_pivot) {
    int i;
    for(i=0; i<k && pivot[i]==last_pivot[i]; ++i){}
    return i;
}

// Calculate sum of distance while combining different pivots. Complexity : O( n^2 )
double SumDistance(const int k, const int n, const int dim, double* coord, int* pivots, int* last_pivots, double* rebuiltCoord){
    int i, k_start;
    if(last_pivots[0]==-1) k_start = 0;
    else k_start = PivotCompare(k, pivots, last_pivots);
    // Rebuild coordinates. New coordinate of one point is its distance to each pivot.
    #pragma omp parallel for proc_bind(close)
    for(i=0; i<n; i++){
        int ki;
        for(ki=k_start; ki<k; ki++){
            double distance = 0;
            int pivoti = pivots[ki];
            int j;
            for(j=0; j<dim; j++){
                distance += pow(coord[pivoti*dim + j] - coord[i*dim + j] ,2);
            }
            rebuiltCoord[i*k + ki] = sqrt(distance);
        }
    }

    // Calculate the sum of Chebyshev distance with rebuilt coordinates between every points
    double chebyshevSum = 0;
    #pragma omp parallel for reduction(+:chebyshevSum) proc_bind(close)
    for(i=0; i<n; i++){
        int j;
        for(j=i; j<n; j++){
            double chebyshev = 0;
            int ki;
            for(ki=0; ki<k; ki++){
                double dis = fabs(rebuiltCoord[i*k + ki] - rebuiltCoord[j*k + ki]);
                chebyshev = dis>chebyshev ? dis : chebyshev;
            }
            chebyshevSum += 2 * chebyshev;
        }
    }

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

void MergeSort(int M, int k, double* distance1, double* distance2, double* distance, int* pivots1, int* pivots2, int* pivots, int order){
    int i1=0, i2=0, i=0;
    int kj;
    // As M=M1=M2, definitely i1<=i<M=M1, i2<=i<M=M2
    while(i<M){
        if(distance1[i1]<=distance2[i2] && order==MIN ||
         distance1[i1]>=distance2[i2] && order==MAX){
            distance[i] = distance1[i1];
            for(kj=0; kj<k; ++kj){
                pivots[i*k + kj] = pivots1[i1*k + kj];
            }
            i1++;
        }
        else{
            distance[i] = distance2[i2];
            for(kj=0; kj<k; ++kj){
                pivots[i*k + kj] = pivots2[i2*k + kj];
            }
            i2++;
        }
        i++;
    }
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
                double* coord, int* pivots, int* maxDisSumPivots, double* maxDistanceSum, int* minDisSumPivots, double* minDistanceSum){
        int i;
        int* last_pivots = (int*)malloc(sizeof(int) * k);
        double* rebuiltCoord = (double*)malloc(sizeof(double) * n * k);
        for(i=0; i<k; ++i) {
            pivots[i] = i;
            last_pivots[i] = -1;
        }

        for(i=0; i<rank; ++i) {
            if(!nextCombination(n-1, k, pivots)) return;
        }

        while(1){
            // Calculate sum of distance while combining different pivots.
            double distanceSum = SumDistance(k, n, dim, coord, pivots, last_pivots, rebuiltCoord);

            // put data at the end of array
            maxDistanceSum[M] = distanceSum;
            minDistanceSum[M] = distanceSum;
            int kj;
            for(kj=0; kj<k; kj++){
                maxDisSumPivots[M*k + kj] = pivots[kj];
                minDisSumPivots[M*k + kj] = pivots[kj];
            }

            // sort
            int a;
            for(a=M; a>0; a--){
                if(maxDistanceSum[a] > maxDistanceSum[a-1]){
                    double temp = maxDistanceSum[a];
                    maxDistanceSum[a] = maxDistanceSum[a-1];
                    maxDistanceSum[a-1] = temp;
                    int kj;
                    for(kj=0; kj<k; kj++){
                        int temp = maxDisSumPivots[a*k + kj];
                        maxDisSumPivots[a*k + kj] = maxDisSumPivots[(a-1)*k + kj];
                        maxDisSumPivots[(a-1)*k + kj] = temp;
                    }
                }
                if(minDistanceSum[a] < minDistanceSum[a-1]){
                    double temp = minDistanceSum[a];
                    minDistanceSum[a] = minDistanceSum[a-1];
                    minDistanceSum[a-1] = temp;
                    int kj;
                    for(kj=0; kj<k; kj++){
                        int temp = minDisSumPivots[a*k + kj];
                        minDisSumPivots[a*k + kj] = minDisSumPivots[(a-1)*k + kj];
                        minDisSumPivots[(a-1)*k + kj] = temp;
                    }
                }
            }

            /** Iteration Log : pivots computed, best pivots, max distance sum, min distance sum pivots, min distance sum
            *** You can delete the logging code. **/
            // int kj;
            // for(kj=0; kj<k; kj++){
            //     printf("%d ", pivots[kj]);
            // }
            // printf("%lf\n", distanceSum);
            // putchar('\t');
            // for(kj=0; kj<k; kj++){
            //     printf("%d ", maxDisSumPivots[kj]);
            // }
            // printf("%lf\t", maxDistanceSum[0]);
            // for(kj=0; kj<k; kj++){
            //     printf("%d ", minDisSumPivots[kj]);
            // }
            // printf("%lf\n", minDistanceSum[0]);

            for(i=0; i<size; ++i){
                if(!nextCombination(n-1, k, pivots)) return;
            }
        }

        free(last_pivots);
        free(rebuiltCoord);
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

    omp_set_num_threads(12);

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
    // Broadcast the coordinate
    MPI_Bcast(coord, n*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Start timing
    struct timeval start;
    gettimeofday(&start, NULL);

    int i;
    // maxDistanceSum : the largest M distance sum
    double* maxDistanceSum = (double*)malloc(sizeof(double) * (M+1));
    // maxDisSumPivots : the top M pivots combinations
    int* maxDisSumPivots = (int*)malloc(sizeof(int) * k * (M+1));
    // minDistanceSum : the smallest M distance sum
    double* minDistanceSum = (double*)malloc(sizeof(double) * (M+1));
    // minDisSumPivots : the bottom M pivots combinations
    int* minDisSumPivots = (int*)malloc(sizeof(int) * k * (M+1));
    
    for(i=0; i<M; i++){
        maxDistanceSum[i] = 0;
        minDistanceSum[i] = __DBL_MAX__;
        int ki;
        for(ki=0; ki<k; ki++){
            maxDisSumPivots[i*k + ki] = 0;
            minDisSumPivots[i*k + ki] = 0;
        }
    }

    // Used for merge sorting
    double* maxDistanceSum1 = (double*)malloc(sizeof(double) * M);
    double* maxDistanceSum2 = (double*)malloc(sizeof(double) * M);
    int* maxDisSumPivots1 = (int*)malloc(sizeof(int) * k * M);
    int* maxDisSumPivots2 = (int*)malloc(sizeof(int) * k * M);
    double* minDistanceSum1 = (double*)malloc(sizeof(double) * M);
    double* minDistanceSum2 = (double*)malloc(sizeof(double) * M);
    int* minDisSumPivots1 = (int*)malloc(sizeof(int) * k * M);
    int* minDisSumPivots2 = (int*)malloc(sizeof(int) * k * M);

    // temp : indexes of pivots with dummy array head
    int* temp = (int*)malloc(sizeof(int) * k);

    // Main loop. Combine different pivots with recursive function and evaluate them. Complexity : O( n^(k+2) )
    Combination(k, n, dim, M, size, rank, coord, temp, maxDisSumPivots, maxDistanceSum, minDisSumPivots, minDistanceSum);

    /* sort
     * rank_data: 2->0, 3->1, 1->0
     */
    switch(rank){
        case 0:
        // 2 -> 0
        MPI_Recv(maxDisSumPivots1, M * k, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(minDisSumPivots1, M * k, MPI_INT, 2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(maxDistanceSum1, M, MPI_DOUBLE, 2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(minDistanceSum1, M, MPI_DOUBLE, 2, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // sort(0, 2)
        MergeSort(M, k, maxDistanceSum, maxDistanceSum1, maxDistanceSum2, 
            maxDisSumPivots, maxDisSumPivots1, maxDisSumPivots2, MAX);
        MergeSort(M, k, minDistanceSum, minDistanceSum1, minDistanceSum2, 
            minDisSumPivots, minDisSumPivots1, minDisSumPivots2, MIN);

        // 1 -> 0
        MPI_Recv(maxDisSumPivots1, M * k, MPI_INT, 1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(minDisSumPivots1, M * k, MPI_INT, 1, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(maxDistanceSum1, M, MPI_DOUBLE, 1, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(minDistanceSum1, M, MPI_DOUBLE, 1, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // sort(0, 1)
        MergeSort(M, k, maxDistanceSum1, maxDistanceSum2, maxDistanceSum, 
            maxDisSumPivots1, maxDisSumPivots2, maxDisSumPivots, MAX);
        MergeSort(M, k, minDistanceSum1, minDistanceSum2, minDistanceSum, 
            minDisSumPivots1, minDisSumPivots2, minDisSumPivots, MIN);
        break;

        case 1:
        // 3 -> 1
        MPI_Recv(maxDisSumPivots1, M * k, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(minDisSumPivots1, M * k, MPI_INT, 3, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(maxDistanceSum1, M, MPI_DOUBLE, 3, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(minDistanceSum1, M, MPI_DOUBLE, 3, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // sort(1, 3)
        MergeSort(M, k, maxDistanceSum, maxDistanceSum1, maxDistanceSum2, 
            maxDisSumPivots, maxDisSumPivots1, maxDisSumPivots2, MAX);
        MergeSort(M, k, minDistanceSum, minDistanceSum1, minDistanceSum2, 
            minDisSumPivots, minDisSumPivots1, minDisSumPivots2, MIN);

        // 1 -> 0
        MPI_Send(maxDisSumPivots2, M * k, MPI_INT, 0, 4, MPI_COMM_WORLD);
        MPI_Send(minDisSumPivots2, M * k, MPI_INT, 0, 5, MPI_COMM_WORLD);
        MPI_Send(maxDistanceSum2, M, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
        MPI_Send(minDistanceSum2, M, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
        break;

        case 2:
        // 2 -> 0
        MPI_Send(maxDisSumPivots, M * k, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(minDisSumPivots, M * k, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(maxDistanceSum, M, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        MPI_Send(minDistanceSum, M, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
        break;

        case 3:
        // 3 -> 1
        MPI_Send(maxDisSumPivots, M * k, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Send(minDisSumPivots, M * k, MPI_INT, 1, 1, MPI_COMM_WORLD);
        MPI_Send(maxDistanceSum, M, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
        MPI_Send(minDistanceSum, M, MPI_DOUBLE, 1, 3, MPI_COMM_WORLD);
        break;
    }

    MPI_Barrier(MPI_COMM_WORLD);
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
        int ki;
        if(rank==0){
            printf("max : ");
            for(ki=0; ki<k; ki++){
                printf("%d ", maxDisSumPivots[ki]);
            }
            printf("%lf\n", maxDistanceSum[0]);

            printf("min : ");
            for(ki=0; ki<k; ki++){
                printf("%d ", minDisSumPivots[ki]);
            }
            printf("%lf\n", minDistanceSum[0]);

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
