
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

int main (void) {
	// Allocate pointers for host and device memory
	float *h_input; // host
	float *d_input; // device

	// Define sample size for mean and std
	const int SAMPLE_SIZE = 50;

	// malloc the host memory
	h_input = (float*) malloc(sizeof(float));
	// cudamalloc the device memory
	cudaMalloc((void**) &d_input, sizeof(float));

	/* START GPU CODE */
	// Declare variable
	curandGenerator_t gen;
	// Create RNG
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	// Set the generator options
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	// Generator the randoms
	// This will create SAMPLE_SIZE random numbers and put it into d_input
	curandGenerateNormal(gen, d_input, SAMPLE_SIZE, 0.0f, 1.0f);
	/* END GPU CODE */

	// send from device to host
	cudaMemcpy(h_input, d_input, sizeof(float), cudaMemcpyDeviceToHost);

	// Initialise placeholder sum
	float sum = 0;

	// Compute sum of all numbers
	for (int i = 0; i < SAMPLE_SIZE; i++) {
		sum += d_input[i];
	}

	// Find mean
	float mean = sum / SAMPLE_SIZE;

	// Find std
	// Find (Xi - mean)^2 for each data point
	float diff_to_mean_squareds[SAMPLE_SIZE];
	for (int i = 0; i < SAMPLE_SIZE; i++) {
		float diff_to_mean = d_input[i] - mean;
		float diff_to_mean_squared = diff_to_mean * diff_to_mean;
		diff_to_mean_squareds[i] = diff_to_mean_squared;
	}
	// Sum all (Xi - mean)^2
	float sum_diff_to_mean_squareds = 0;
	for (int i = 0; i < SAMPLE_SIZE; i++) {
		sum_diff_to_mean_squareds += diff_to_mean_squareds[i];
	}

	// Divide by sample size
	float var = sum_diff_to_mean_squareds / SAMPLE_SIZE;

	// Finally, take square root of var to get std
	float std = sqrtf(var);

	printf("mean is %f, std is %f", mean, std);

	return(0);
}
