#include <iostream>

#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
//#include <thrust/fill.h>

// examples
// [1] https://thrust.github.io/doc/group__transformed__reductions_ga0d4232a9685675f488c3cc847111e48d.html 


template <typename T>
struct summary_stats_data
{
    T n, mean, M2; // min, max,
    void initialize()
    {
      n = mean = M2 = 0;
    }
    T stdev()      { return std::sqrt( M2 / n ); }
};


// stats_unary_op is a functor that takes in a value x and
// returns a variace_data whose mean value is initialized to x.
template <typename T>
struct summary_stats_unary_op
{
    __host__ __device__
    summary_stats_data<T> operator()(const T& x) const
    {
         summary_stats_data<T> result;
         result.n    = 1;
         result.mean = x;
         result.M2   = 0;
         return result;
    }
};

// summary_stats_binary_op is a functor that accepts two summary_stats_data
// structs and returns a new summary_stats_data which are an
// approximation to the summary_stats for
// all values that have been agregated so far
template <typename T>
struct summary_stats_binary_op
    : public thrust::binary_function<const summary_stats_data<T>&,
                                     const summary_stats_data<T>&,
                                           summary_stats_data<T> >
{
    __host__ __device__
    summary_stats_data<T> operator()(const summary_stats_data<T>& x, const summary_stats_data <T>& y) const
    {
        summary_stats_data<T> result;

        // precompute some common subexpressions
        T n  = x.n + y.n;
        T delta  = y.mean - x.mean;
        T delta2 = delta  * delta;

        //Basic number of samples (n), min, and max
        result.n   = n;
        result.mean = x.mean + delta * y.n / n;
        result.M2  = x.M2 + y.M2;
        result.M2 += delta2 * x.n * y.n / n;

        return result;
    }
};



void thrust_transform_reduce(thrust::device_vector<float> &d_edge_buf,
								int bcols,
								int bsize);

void my_version(thrust::device_vector<float> &d_edge_buf,
								int bcols,
								int bsize);

__global__ void my_transform_reduce(float *g_input,
		float *g_out_mean,
		float *g_out_stdev,
		const int len,
		const int warpNum
		);


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
	std::cout << "initialize data on the host\n";

	int bcols = 3;
	int bsize = 4;

	int edge_buf_size = bcols * bsize;

	thrust::host_vector<float>   h_edge_buf(edge_buf_size);

	for(int row=0; row<bsize; row++) { 
		for(int col=0; col<bcols; col++) { 
			h_edge_buf[row * bcols + col] = (float)(col + 1);
			std::cout << h_edge_buf[row * bcols + col] << " "; 
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	//-----------------------------------------------------------------------//
	std::cout << "\ncopy host to device\n";
	thrust::device_vector<float>     d_edge_buf=h_edge_buf;

	//-----------------------------------------------------------------------//
	std::cout << "\ntesting thrust::transform_reduce\n";
	thrust_transform_reduce(d_edge_buf, bcols, bsize);

	//-----------------------------------------------------------------------//
	std::cout << "\ntesting customized version \n";
	my_version(d_edge_buf, bcols, bsize);

	return 0;
}

void thrust_transform_reduce(thrust::device_vector<float> &d_edge_buf,
								int bcols,
								int bsize)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float local_ms = 0.f;

	summary_stats_unary_op<float> 	unary_op;
	summary_stats_binary_op<float> 	binary_op;
	summary_stats_data<float>	stat_init;
	summary_stats_data<float>	stat_result;

	thrust::host_vector<float>   h_means(1,0.f);
	thrust::host_vector<float>   h_stdevs(1,0.f);

	stat_init.initialize();

	cudaEventRecord(start, 0);

	// gpu code
	stat_result = thrust::transform_reduce(d_edge_buf.begin(),
										   d_edge_buf.end(),
										   unary_op,
										   stat_init,
										   binary_op);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&local_ms, start, stop);

	printf("(thrust::transform_reduce) runtime = %lf (ms)\n", local_ms);

	h_means[0] = stat_result.mean;
	h_stdevs[0] = stat_result.stdev();

	std::cout << "means = " << h_means[0] << std::endl; 
	std::cout << "stdev = " << h_stdevs[0] << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}



void my_version(thrust::device_vector<float> &d_edge_buf,
								int bcols,
								int bsize)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float local_ms = 0.f;

	thrust::device_vector<float>   d_means(1);
	thrust::device_vector<float>   d_stdevs(1);

	thrust::host_vector<float>   h_means(1,0.f);
	thrust::host_vector<float>   h_stdevs(1,0.f);

	float *g_input       = thrust::raw_pointer_cast(d_edge_buf.data());
	float *g_out_mean    = thrust::raw_pointer_cast(d_means.data());
	float *g_out_stdev   = thrust::raw_pointer_cast(d_stdevs.data());

	int len = bcols * bsize;
	int warpNum = ( len + 31 ) / 32;
	int Blk = warpNum * 32;

	cudaEventRecord(start, 0);

	my_transform_reduce <<< 1, Blk >>> (g_input,
										g_out_mean,
										g_out_stdev,
										len,
										warpNum
										); 

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&local_ms, start, stop);

	printf("(my version) runtime = %lf (ms)\n", local_ms);

	h_means = d_means;
	h_stdevs= d_stdevs;

	//---------------------//
	// check output
	//---------------------//
	std::cout << "\noutput vals\n";
	std::cout << "means = " << h_means[0] << std::endl; 
	std::cout << "stdev = " << h_stdevs[0] << std::endl; 
}



__global__ void my_transform_reduce(float *g_input,
		float *g_out_mean,
		float *g_out_stdev,
		const int len,
		const int warpNum
		)
{
	__shared__ float sm_val[32]; // max 32 waps
	__shared__ float sm_d2[32];
	__shared__ float sm_mean;

	int lid = threadIdx.x;

	//--------------//
	// compute mean
	//--------------//
	float v = 0.f;
	float in_x = 0.f;

	if(lid < len) {
		in_x = v = g_input[lid]; // v for reduction, in_x for later use
	}

	int lane_id = lid % 32; 
	int warp_id = lid / 32;

	for(int i= 16; i>0; i>>=1) {
		v += __shfl_down_sync(0xFFFFFFFF, v, i, 32);
	}

	if(lane_id == 0) {
		sm_val[warp_id]	= v;
	}

	__syncthreads();

	if(lid == 0) {
		float sum = 0.f;
		for(int i=0;i<warpNum;i++) {
			sum += sm_val[i];		
		}

		float result_mean = sum / float(len);
		g_out_mean[0] = result_mean;  // compute mean
		sm_mean = result_mean;  // update mean to shared mem
	}

	__syncthreads();

	//--------------//
	// compute stdev
	//--------------//
	float diff2 = 0.f;

	if(lid < len) {
		float dif = in_x - sm_mean; // xi - u
		diff2 = dif * dif; 
	}

	for(int i= 16; i>0; i>>=1) {
		diff2 += __shfl_down_sync(0xFFFFFFFF, diff2, i, 32);
	}

	if(lane_id == 0) {
		sm_d2[warp_id] = diff2;
	}

	__syncthreads();

	if(lid == 0) {
		float stdev = 0.f;
		for(int i=0;i<warpNum;i++) {
			stdev += sm_d2[i];		
		}
		g_out_stdev[0] = sqrtf(stdev / float(len));
	}

}
