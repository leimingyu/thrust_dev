#include <iostream>

#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

// examples
// [1] https://github.com/thrust/thrust/wiki/Quick-Start-Guide

void thrust_reduce_by_key(thrust::device_vector<int> &d_keys,
		                  thrust::device_vector<float> &d_array,
						  const int Ksize, const int Len);

void reduce_by_key_v1(thrust::device_vector<int> &d_keys,
		              thrust::device_vector<int> &d_norep_keys,
		              thrust::device_vector<float> &d_array,
					  const int Ksize, const int Len);

__global__ void reduce_by_key_kernel_v1 (int* g_keys,
		int* g_norep_keys,
		float* g_input,
		const int Ksize,
		const int Len,
		float* g_output);

int main(void)
{
	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
	printf("Device name: %s\n", prop.name);
	cudaSetDevice(0);


	int major = THRUST_MAJOR_VERSION;
	int minor = THRUST_MINOR_VERSION;
	std::cout << "Thrust v" << major << "." << minor << std::endl;

	//-----------------------------------------------------------------------//
	std::cout << "initialize keys on the host\n";
	const int Len = 896;
	const int Ksize = 124;

	thrust::host_vector<int>   h_keys(Len);
	thrust::host_vector<float> h_array(Len);

	for(int i=0; i<Len; i++) { 
		int ii = i / 7;
		int k = Ksize - (ii + 1); // repeat the value 7 times 
		if(k <0) k =0;

		h_keys[i] = k;
		h_array[i] = (float) k;

		//std::cout << h_keys[i] << " ";
	}
	//std::cout << std::endl;

	//-----------------------------------------------------------------------//
	std::cout << "no-repeat keys in order\n";

	thrust::host_vector<int>   h_norep_keys;
	int prev, curr;
	for(int i=0; i<Len; i++) { 
		if(i==0) {
			prev = h_keys[i];
			h_norep_keys.push_back(prev);	
			continue;
		}

		curr = h_keys[i];

		if(curr != prev) {
			h_norep_keys.push_back(curr);	
			prev = curr;
		}
	}

	std::cout << "no repeat keys size = " << h_norep_keys.size() << std::endl; 

	if(h_norep_keys.size() != Ksize) {
		std::cerr << "Ksize is not equal to norep_keys.size()\n"; 
		return -1;	
	}

	for(int i=0; i<h_norep_keys.size(); i++) { 
		std::cout << h_norep_keys[i] << " ";
	}
	std::cout << std::endl;


	//-----------------------------------------------------------------------//
	std::cout << "\ncopy host to device\n";

	thrust::device_vector<int>     d_keys=h_keys;
	thrust::device_vector<int>     d_norep_keys=h_norep_keys;
	thrust::device_vector<float>   d_array=h_array;

	//-----------------------------------------------------------------------//
	std::cout << "\ntesting thrust::reduce_by_key\n";
	thrust_reduce_by_key(d_keys, d_array, Ksize, Len);


	//-----------------------------------------------------------------------//
	std::cout << "\ntesting customized reduce_by_key (v1)\n";
	reduce_by_key_v1(d_keys, d_norep_keys, d_array, Ksize, Len);

	return 0;
}

void thrust_reduce_by_key(thrust::device_vector<int> &d_keys,
		                  thrust::device_vector<float> &d_array,
						  const int Ksize, const int Len)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float local_ms;
	float gputime_ms = 0.f;
	
	// initialize output 
	thrust::device_vector<int>   d_out_keys(Ksize, 0);
	thrust::device_vector<float> d_out_vals(Ksize, 0.f);

	for(int reps=0; reps<100; reps++)
	{
		local_ms = 0.f;

		cudaEventRecord(start, 0);

		thrust::reduce_by_key(d_keys.begin(),
				d_keys.end(),
				d_array.begin(),
				d_out_keys.begin(),
				d_out_vals.begin()
				);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&local_ms, start, stop);
		gputime_ms += local_ms;
	}

	printf("(thrust::reduce_by_key) runtime = %lf (ms)\n", gputime_ms * 0.01);


	thrust::host_vector<int>   h_out_keys;
	thrust::host_vector<float> h_out_vals;

	h_out_keys = d_out_keys;
	h_out_vals = d_out_vals;

	//---------------------//
	// check output
	//---------------------//

	std::cout << "\noutput keys\n";
	for(int i=0; i<h_out_keys.size(); i++) { 
		std::cout << h_out_keys[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "\noutput vals\n";
	for(int i=0; i<h_out_vals.size(); i++) { 
		std::cout << h_out_vals[i] << " ";
	}
	std::cout << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

__global__ void reduce_by_key_kernel_v1 (int* g_keys,
		int* g_norep_keys,
		float* g_input,
		const int Ksize,
		const int Len,
		float* g_output)
{
	__shared__ int   sm_keys[256];
	__shared__ float sm_vals[256];

	int lid = threadIdx.x;

	if(lid < Ksize) { // 124
		sm_keys[lid] = g_norep_keys[lid];	
		sm_vals[lid] = 0.f;
	}

	__syncthreads();


	if(lid < Len) {
		int   my_key = g_keys[lid];
		float my_val = g_input[lid];

		for(int i=0; i<Ksize; i++) {
			if(my_key == sm_keys[i]) {
				atomicAdd(&sm_vals[i], my_val);
				break;
			}
		}
	}

	__syncthreads();

	if(lid < Ksize) {
		g_output[lid] = sm_vals[lid];
	}

}

void reduce_by_key_v1(thrust::device_vector<int> &d_keys,
		              thrust::device_vector<int> &d_norep_keys,
		              thrust::device_vector<float> &d_array,
					  const int Ksize, const int Len)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float local_ms;
	float gputime_ms = 0.f;

	
	// initialize output 
	thrust::device_vector<float> d_out_vals(Ksize, 0.f);


	int   *g_keys        = thrust::raw_pointer_cast(d_keys.data());
	int   *g_norep_keys  = thrust::raw_pointer_cast(d_norep_keys.data());
	float *g_input       = thrust::raw_pointer_cast(d_array.data());
	float *g_output      = thrust::raw_pointer_cast(d_out_vals.data());

	if(Len > 1024) {
		printf("(Bummer) => I need to work on this configuration\n");
		exit(1);
	}else{
		dim3 Grds(1,1,1);
		dim3 Blks(Len,1,1);


		// Measure the runtime
		for(int reps=0; reps<100; reps++)
		{
			local_ms = 0.f;
			cudaEventRecord(start, 0);

			// NOTE: need to write a template for customization
			reduce_by_key_kernel_v1 <<< Grds, Blks >>> (g_keys,
					g_norep_keys,
					g_input,
					Ksize,
					Len,
					g_output
					); 

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&local_ms, start, stop);
			gputime_ms += local_ms;
		}

		printf("(reduce_by_key_v1) runtime = %lf (ms)\n", gputime_ms * 0.01);
	}

	thrust::host_vector<float> h_out_vals;
	h_out_vals = d_out_vals;

	//---------------------//
	// check output
	//---------------------//
	std::cout << "\noutput vals\n";
	for(int i=0; i<h_out_vals.size(); i++) { 
		std::cout << h_out_vals[i] << " ";
	}
	std::cout << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
