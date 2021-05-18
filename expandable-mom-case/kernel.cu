#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define CUSTOMMERS 400
#define PEOPLE_PER_TABLE 10
#define WAITERS_PER_TABLE 2
#define TABLES CUSTOMMERS/PEOPLE_PER_TABLE

using namespace std;

// GPU function that simulates an sleep function
__device__ void wait_gpu(float time) {
	int wait = 1000000 * time * 2.5;
	for (int i = 0; i < wait; i++)
	{
	}
}

// GPU function that simulates the serving dishes time
__global__ void GPU_serving_dishes(int round_offset, bool* plates) 
{
	int gId = threadIdx.x + round_offset;
	wait_gpu(1);
	plates[gId] = true;
}

// GPU function that simulates the eating and picking plates up time
__global__ void GPU_finish_eating_and_picking_up_plates(bool* plates, float* eating_times, clock_t* global_now)
{
	int gId = threadIdx.x;
	wait_gpu(eating_times[gId]);
	wait_gpu(1);
	plates[gId] = true;
}

// Simulates how the catering service will behave in the event
__host__ void CPU_serving_dishes(char food_course[], cudaEvent_t s, cudaEvent_t e)
{
	printf("SERVICE: serving %s ...\n", food_course);

	// Determine how many rounds did the waiters will need to do, to serve a table
	int serving_rounds = CUSTOMMERS / (TABLES * WAITERS_PER_TABLE);

	// Declare host and device variables
	bool* host_customers_plate;
	bool* dev_customers_plate;

	// Reserve space for host and device variables
	host_customers_plate = (bool*)malloc(CUSTOMMERS * sizeof(bool));
	cudaMalloc((void**)&dev_customers_plate, CUSTOMMERS * sizeof(bool));

	// Initialice host variables
	for (int i = 0; i < CUSTOMMERS; i++)
	{
		host_customers_plate[i] = false;
	}
	float totalTime = 0;

	// Start serving rounds
	for (int i = 0; i < serving_rounds; i++)
	{
		cudaMemcpy(dev_customers_plate, host_customers_plate, CUSTOMMERS * sizeof(bool), cudaMemcpyHostToDevice);
		dim3 block(CUSTOMMERS / serving_rounds);

		// Start kernel function and capture the time it spend in the execution
		cudaEventCreate(&s);
		cudaEventCreate(&e);
		cudaEventRecord(s, 0);
		GPU_serving_dishes << <1, block >> > (i * CUSTOMMERS / serving_rounds, dev_customers_plate);
		cudaEventRecord(e, 0);
		cudaDeviceSynchronize();
		float currElapsedTime;
		cudaEventElapsedTime(&currElapsedTime, s, e);
		totalTime += currElapsedTime;
		cudaEventDestroy(s);
		cudaEventDestroy(e);
		
		cudaMemcpy(host_customers_plate, dev_customers_plate, CUSTOMMERS * sizeof(bool), cudaMemcpyDeviceToHost);

		// Calculate the percentage of people that have been served
		float percentage_served = 0;
		int persons_served = 0;
		for (int i = 0; i < CUSTOMMERS; i++)
		{
			if (host_customers_plate[i]) {
				persons_served++;
			}
		}
		percentage_served = 100 * persons_served / (float)CUSTOMMERS;
		printf("SERVICE: people served -> %.2f%%\n", percentage_served);

	}

	// Free space for host and device variables
	free(host_customers_plate);
	cudaFree(dev_customers_plate);
	printf("SERVICE: %s served in %.3f ms\n", food_course, totalTime);
}

// Generates a random number from 0-10 
__host__ float getRangeRandom()
{
	return (rand() % 11 / 10.0) * 10;
}

// Fills a float list with random numbers
__host__ void fillRandomNumbersList(float* randomNumbersList, int size)
{
	for (int i = 0; i < size; i++)
	{
		randomNumbersList[i] = getRangeRandom()+1;
	}
}

// Simulates the eating and picking up of the plates when a costummer has finish
__host__ void CPU_finish_eating_and_picking_up_plates(cudaEvent_t s, cudaEvent_t e) {
	printf("FOOD: people eating...\n");
	int serving_rounds = CUSTOMMERS / (TABLES * WAITERS_PER_TABLE);

	// Declare host and device variables
	bool* host_customers_plate;
	bool* dev_customers_plate;
	float* host_random_eating_times;
	float* dev_random_eating_times;
	clock_t* host_global_clock;
	clock_t* dev_global_clock;

	// Reserve space for host and device variables
	host_customers_plate = (bool*)malloc(CUSTOMMERS * sizeof(bool));
	host_random_eating_times = (float*)malloc(CUSTOMMERS * sizeof(float));
	host_global_clock = (clock_t*)malloc(sizeof(clock_t));
	cudaMalloc((void**)&dev_customers_plate, CUSTOMMERS * sizeof(bool));
	cudaMalloc((void**)&dev_random_eating_times, CUSTOMMERS * sizeof(float));
	cudaMalloc((void**)&dev_global_clock, sizeof(clock_t));

	// Initialice host variables
	for (int i = 0; i < CUSTOMMERS; i++)
	{
		host_customers_plate[i] = false;
	}
	fillRandomNumbersList(host_random_eating_times, CUSTOMMERS);

	cudaMemcpy(dev_customers_plate, host_customers_plate, CUSTOMMERS * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_random_eating_times, host_random_eating_times, CUSTOMMERS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_global_clock, host_global_clock, sizeof(clock_t), cudaMemcpyHostToDevice);
	dim3 block(CUSTOMMERS);

	// Start kernel function and capture the time it spend in the execution
	float totalTime = 0;
	cudaEventCreate(&s);
	cudaEventCreate(&e);
	cudaEventRecord(s, 0);
	GPU_finish_eating_and_picking_up_plates << < 1, block >> > (dev_customers_plate,dev_random_eating_times, dev_global_clock);
	cudaEventRecord(e, 0);
	cudaDeviceSynchronize();
	float currElapsedTime;
	cudaEventElapsedTime(&currElapsedTime, s, e);
	totalTime += currElapsedTime;
	cudaEventDestroy(s);
	cudaEventDestroy(e);

	cudaMemcpy(host_customers_plate, dev_customers_plate, CUSTOMMERS * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_random_eating_times, dev_random_eating_times, CUSTOMMERS * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_global_clock, dev_global_clock, sizeof(clock_t), cudaMemcpyDeviceToHost);

	// Free space for host and device variables
	free(host_customers_plate);
	free(host_random_eating_times);
	free(host_global_clock);
	cudaFree(dev_customers_plate);
	cudaFree(dev_random_eating_times);
	cudaFree(dev_global_clock);
	printf("FOOD: people finish eating and plates picked up in %.3f ms\n", currElapsedTime);
}

int main() {
	printf("START\n");

	printf("SERVICE: Preparing food...\n");
	_sleep(1000);
	printf("SERVICE: Food ready\n");

	// Starting cuda events, these help us calculating the time spend in the kernel
	cudaEvent_t start;
	cudaEvent_t end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	// Three course dinner
	CPU_serving_dishes("starter", start, end);
	CPU_finish_eating_and_picking_up_plates(start, end);
	CPU_serving_dishes("main course", start, end);
	CPU_finish_eating_and_picking_up_plates(start, end);
	CPU_serving_dishes("dessert", start, end);
	CPU_finish_eating_and_picking_up_plates(start, end);

	printf("END\n");
	return 0;
}