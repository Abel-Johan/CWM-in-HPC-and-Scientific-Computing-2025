#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

#define NUM_ELS 100000

int main() {
    // initialise an array with dynamic memory allocation
    // also initialise a variable to store sum in
    float *random_array;
    float sum=0;

    random_array = (float*) malloc(NUM_ELS * sizeof(float));

    // initialise loop variable
    int i;

    // parallelise
    #pragma omp parallel shared(sum, random_array) private(i)
    {
        #pragma omp for
        for(i=0; i<NUM_ELS; i++) {
            float x = ((float)rand())/((float)RAND_MAX);
            random_array[i] = x;
        }
        #pragma omp for
        for(int i=0; i<NUM_ELS; i++) {
            sum+=random_array[i];
        }
    }
    printf("\nAverage:\t%f\n", sum/(float)NUM_ELS);

    return(0);
}
