#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#define TILE_SIZE 32

__global__ void computeLevel(unsigned long height, unsigned long width, unsigned char *device_inimg, int *device_level)
{
    __shared__ int hist[256];
    __shared__ int sum1;
    __shared__ int sumB;
    __shared__ int total;
    __shared__ int wB;
    if (threadIdx.x < 256)
        hist[threadIdx.x] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int temp, k, wF, maximum, level, mF, val;
    if (i < height && j < width)
    {
        temp = device_inimg[i * width + j];
        atomicAdd(&hist[temp], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        total = 0;
        sum1 = 0;
        for (k = 0; k < 256; k++)
        {
            total += hist[k];
            sum1 += k * hist[k];
        }
        __syncthreads();
        wB = 0;
        sumB = 0;
        maximum = 0;
        level = 0;

        for (k = 0; k < 256; k++)
        {
            wB = wB + hist[k];
            sumB = sumB + (k-1)*hist[k];
            wF = total - wB;
            if (wB > 0 && wF > 0)
            {
                mF = (sum1 - sumB) / wF;
                val = wB * wF * ((sumB / wB) - mF) * ((sumB / wB) - mF);
                if (val >= maximum)
                {
                    level = k;
                    maximum = val;
                }
            }
        }
        __syncthreads();
        *device_level = level;
    }
}

__global__ void thresholdImage(unsigned long height, unsigned long width, unsigned char *device_inimg, int *device_level, unsigned char *device_outimg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int level = *device_level;
    if (i < height && j < width)
    {
        if (device_inimg[i * width + j] < level)
        {
            device_outimg[i * width + j] = 0;
        }
        else
        {
            device_outimg[i * width + j] = 255;
        }
    }
}

void read_rawimage(char *fname, unsigned long length, unsigned long width, unsigned char *image)

{
    short i;
    FILE *file;

    file = fopen(fname, "r");
    for (i = 0; i < length; i++)
        fread(&image[i * width], 1, width, file);
    fclose(file);
}

void write_rawimage(char *fname, unsigned long length, unsigned long width, unsigned char *image)

{
    short i;
    FILE *file;

    file = fopen(fname, "w");
    for (i = 0; i < length; i++)
        fwrite(&image[i * width], 1, width, file);
    fclose(file);
}

int main(int argc, char *argv[])
{
    char infname[50], outfname[50];
    unsigned long height, width;
    unsigned char *inimg;
    unsigned char *outimg;

    // cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (argc < 4)
    {
        printf("usage is: %s inimg height width [outimg]\n", argv[0]);
        exit(-1);
    }
    strcpy(infname, argv[1]);
    height = (unsigned long)atoi(argv[2]);
    width = (unsigned long)atoi(argv[3]);
    strcpy(outfname, argv[4]);

    // total size of inimg and outimg
    size_t size = height * width * sizeof(unsigned char);
    // allocate host memory
    inimg = (unsigned char *)malloc(size);
    outimg = (unsigned char *)malloc(size);
    // allocate in device memory
    unsigned char *device_inimg;
    cudaMalloc((void **)&device_inimg, size);
    unsigned char *device_outimg;
    cudaMalloc((void **)&device_outimg, size);
    int *device_level;
    cudaMalloc((void **)&device_level, sizeof(int));
    // readimage
    read_rawimage(infname, height, width, inimg);
    // time for cuda
    cudaEventRecord(start, 0);
    // copy from host to device
    cudaMemcpy(device_inimg, inimg, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float H2DTime;
    cudaEventElapsedTime(&H2DTime, start, stop);
    printf("MemCpy Host to Dev time %f\n", H2DTime);
    // kernel (set grid and block dimensions)
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((int)ceil((float)width /(float)TILE_SIZE), (int)ceil((float)height /(float)TILE_SIZE));
    cudaEventRecord(start, 0);
    //kernel function to compute the threshold (aka level)
    computeLevel<<<dimGrid, dimBlock>>>(height, width, device_inimg, device_level); 
    cudaDeviceSynchronize();
    //kernel function to produce the new image
    thresholdImage<<<dimGrid, dimBlock>>>(height, width, device_inimg, device_level, device_outimg); 
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float kernelTime;
    cudaEventElapsedTime(&kernelTime, start, stop);
    printf("Kernel time %f\n", kernelTime);
    // copy from device to host
    cudaEventRecord(start, 0);
    cudaMemcpy(outimg, device_outimg, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // stoptime
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float D2HTime;
    cudaEventElapsedTime(&D2HTime, start, stop);
    printf("MemCpy Dev to Host time %f\n", D2HTime);
    // display the timing results
    printf("Time for image otsu thresholding method. MemCopy %3.2f: Kernel %3.3f: Total %3.2f ms\n", H2DTime + D2HTime, kernelTime, H2DTime + D2HTime + kernelTime);
    // writeimage
    write_rawimage(outfname, height, width, outimg);
    // free device memory
    cudaFree(device_inimg);
    cudaFree(device_outimg);
    cudaFree(device_level);
    // destroy events to free memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // free hostage memory
    free(outimg);
    free(inimg);

    return 0;
}
