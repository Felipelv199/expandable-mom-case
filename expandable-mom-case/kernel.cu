#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define CUSTOMMERS 400
#define PEOPLE_PER_TABLE 10
#define WAITERS_PER_TABLE 2
#define TABLES CUSTOMMERS/PEOPLE_PER_TABLE

using namespace std;

__global__ void GPU_serving_dishes(int round_offset, bool* plates) 
{
	int gId = threadIdx.x + round_offset;
	plates[gId] = true;
}

__host__ void CPU_serving_dishes() 
{
	int serving_rounds = CUSTOMMERS / (TABLES * WAITERS_PER_TABLE);

	bool* host_customers_plate;
	bool* dev_customers_plate;

	host_customers_plate = (bool*)malloc(CUSTOMMERS * sizeof(bool));
	cudaMalloc((void**)&dev_customers_plate, CUSTOMMERS * sizeof(bool));

	for (int i = 0; i < CUSTOMMERS; i++)
	{
		host_customers_plate[i] = false;
	}
	for (int i = 0; i < serving_rounds; i++)
	{
		cudaMemcpy(dev_customers_plate, host_customers_plate, CUSTOMMERS * sizeof(bool), cudaMemcpyHostToDevice);
		dim3 block(CUSTOMMERS / serving_rounds);
		GPU_serving_dishes << <1, block >> > (i * CUSTOMMERS / serving_rounds, dev_customers_plate);
		cudaDeviceSynchronize();
		cudaMemcpy(host_customers_plate, dev_customers_plate, CUSTOMMERS * sizeof(bool), cudaMemcpyDeviceToHost);

		float percentage_served = 0;
		int persons_served = 0;
		for (int i = 0; i < CUSTOMMERS; i++)
		{
			if (host_customers_plate[i]) {
				persons_served++;
			}
		}
		percentage_served = 100 * persons_served / (float)CUSTOMMERS;
		printf("Persons served: %.2f%%\n", percentage_served);

	}

	free(host_customers_plate);
	cudaFree(dev_customers_plate);
}

__host__ float getRangeRandom()
{
	return (rand() % 11 / 10.0) * 10;
}


__host__ void fillRandomNumbersList(float* randomNumbersList, int size)
{
	for (int i = 0; i < size; i++)
	{
		randomNumbersList[i] = getRangeRandom();
	}
}

__host__ void CPU_cleaning_table_and_picking_up_plates() {
	int serving_rounds = CUSTOMMERS / (TABLES * WAITERS_PER_TABLE);

	bool* host_customers_plate;
	bool* dev_customers_plate;
	float* host_random_eating_times;
	float* dev_random_eating_times;

	host_customers_plate = (bool*)malloc(CUSTOMMERS * sizeof(bool));
	host_random_eating_times = (float*)malloc(CUSTOMMERS * sizeof(float));
	cudaMalloc((void**)&dev_customers_plate, CUSTOMMERS * sizeof(bool));
	cudaMalloc((void**)&dev_random_eating_times, CUSTOMMERS * sizeof(float));

	for (int i = 0; i < CUSTOMMERS; i++)
	{
		host_customers_plate[i] = false;
	}
	fillRandomNumbersList(host_random_eating_times, CUSTOMMERS);

	cudaMemcpy(dev_customers_plate, host_customers_plate, CUSTOMMERS * sizeof(bool), cudaMemcpyHostToDevice);

	free(host_customers_plate);
	cudaFree(dev_customers_plate);
}
int main() {
	printf("START\n");

	printf("Cooking food...\n");
	_sleep(1000);
	printf("Food Cooked\n");

	printf("Serving food...\n");
	CPU_serving_dishes();
	printf("Food served\n");
	/*
	printf("People Eating...\n");
	CPU_cleaning_table_and_picking_up_plates();
	printf("Some People Finishing\n");
	*/
	return 0;
}