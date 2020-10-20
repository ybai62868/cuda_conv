#include <cstdio>


int getThreadNum()
{
	cudaDeviceProp prop;
	int count;
	cudaGetDeviceCount(&count);
	printf("gpu num %d\n", count);

	cudaGetDeviceProperties(&prop, 0);
	printf("max thread num : %d\n", prop.maxThreadsPerBlock);
	printf("grid dimensions : %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	return prop.maxThreadsPerBlock;
}

__global__ void conv (float* img, float* kernel, float* result, int width, int
		height, int channel, int kernelSize)
	
{
	int ti = threadIdx.x;
	int bi = blockIdx.x;
	// int id = (bi * threadNum + ti);
	int id = (bi * blockDim.x + ti);
	if (id >= width * height * channel) return;
	
	int row = id / width;
	int col = id % width;

	for ( int k = 0;k < channel;++k ) {
		for ( int i = 0;i < kernelSize;++i ) {
			for ( int j = 0;j < kernelSize;++j ) {
				float imgValue = 0.0;
				int curRow = row - kernelSize >> 1 + i;
				int curCol = col - kernelSize >> 1 + j;
				if ( curRow < 0 || curCol < 0 || curRow >= height || curCol >= width) {
						continue;
				} else {
					imgValue = img[curRow * width + curCol];
				}
				result[id] += kernel[k * kernelSize * kernelSize + i * kernelSize + j] * imgValue;
			}
		}
	}
//	for ( int i = 0;i < kernelSize; ++i ) {
//		for ( int j = 0;j < kernelSize; ++j ) {
//			float imgValue = 0.0;
//			int curRow = row - kernelSize / 2 + i;
//			int curCol = col - kernelSize / 2 + j;
//			if (curRow < 0 || curCol < 0 || curRow >= height || curCol >= width) {
//				continue;	
//			} else {
//				imgValue = img[curRow * width + curCol];
//			}
//			result[id] += kernel[i * kernelSize + j] * imgValue;
//		}
//	}  
}


int main(void)
{
	int width = 1920;
	int height = 1080;
	int inChannel = 1;
	int outChannel = 8;

	float* img = new float[width * height];

	for ( int row = 0;row < height;++row ) {
		for (int col = 0;col < width;++col ) {
			img[col + row * width] = (col + row) % 256;
		}

	}		
	int kernelSize = 5;
	float* kernel = new float[outChannel * kernelSize * kernelSize];

	for ( int i = 0;i < outChannel * kernelSize * kernelSize; ++i) {
		kernel[i] = i % kernelSize - 1;
	}

	float* imgGpu;
	float* kernelGpu;
	float* resultGpu;

	cudaMalloc((void**)&imgGpu, inChannel * width * height * sizeof(float));
	cudaMalloc((void**)&kernelGpu, outChannel * kernelSize * kernelSize * sizeof(float));
	cudaMalloc((void**)&resultGpu, outChannel * width * height * sizeof(float));


	cudaMemcpy(imgGpu, img, inChannel * width * height * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(kernelGpu, kernel, outChannel * kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);


	int threadNum = getThreadNum();
	int blockNum = (width * height - 0.5) / threadNum + 1;
	conv<<<blockNum, threadNum>>>(imgGpu, kernelGpu, resultGpu, width, height,
			outChannel, kernelSize);
	

	float* result = new float[outChannel * width * height];
	cudaMemcpy(result, resultGpu, outChannel * width * height * sizeof(float), cudaMemcpyDeviceToHost);
	
	

	for ( int i = 0;i < 10;++i ) {
		for ( int j = 0;j < 10;++j ) {
			printf("%2.0f ", img[i * width + j]);
		}
		printf("\n");
	} 
	printf("\n");

	for ( int k = 0;k < 2;++k ) {
	for ( int i = 0;i < kernelSize;++i ) {
		for ( int j = 0;j < kernelSize;++j ) {
			printf("%2.0f ", kernel[k * kernelSize * kernelSize + i * kernelSize + j]);
		}
		printf("\n");
	}
	printf("\n\n");
	}

	for ( int k = 0;k < 2;++k ){ 
	for ( int i = 0;i < 10;++i ) {
		for ( int j = 0;j < 10;++j ) {
			printf("%2.0f ", result[k * height * width + i * width + j]);
		}
		printf("\n");
	}	
	printf("\n\n");
	}

	return 0;
}
