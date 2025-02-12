#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <CL/cl.h>
#include <chrono>

using namespace cv;

// Constants for image dimensions
const size_t WIDTH = 1280;
const size_t HEIGHT = 960;
const size_t NUM_PIXELS = 1228800;

// Function to read images using OpenCV
float* read_image(const std::string& filepath) {
    Mat image = imread(filepath, IMREAD_UNCHANGED);
    float* data = (float*)malloc(NUM_PIXELS * sizeof(float));
    for (size_t i = 0; i < NUM_PIXELS; i++) {
        data[i] = static_cast<float>(image.at<ushort>(i));
    }
    return data;
}

// OpenCL kernel for NDVI calculation
const char* kernelSource = 
"_kernel void calculate_ndvi(_global const float* nir, __global const float* red, __global float* ndvi, const int n) {"
"    int global_id = get_global_id(0);"
"    if (global_id < n) {"
"        float nir_value = nir[global_id];"
"        float red_value = red[global_id];"
"        ndvi[global_id] = ((nir_value + red_value) <= 0) ? 0 : (nir_value - red_value) / (nir_value + red_value);"
"    }"
"}";

// Function to process a single image
void process_image(const std::string& nir_filepath, const std::string& red_filepath, const std::string& output_filepath, cl_command_queue queue, cl_kernel kernel, cl_mem bufferNIR, cl_mem bufferRED, cl_mem bufferNDVI) {
    float* nir_image = read_image(nir_filepath);
    float* red_image = read_image(red_filepath);
    
    clEnqueueWriteBuffer(queue, bufferNIR, CL_TRUE, 0, NUM_PIXELS * sizeof(float), nir_image, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferRED, CL_TRUE, 0, NUM_PIXELS * sizeof(float), red_image, 0, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferNIR);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferRED);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferNDVI);
    clSetKernelArg(kernel, 3, sizeof(int), &NUM_PIXELS);

    size_t globalSize = NUM_PIXELS;
    size_t localSize = 256;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    float* ndvi_result = (float*)malloc(NUM_PIXELS * sizeof(float));
    clEnqueueReadBuffer(queue, bufferNDVI, CL_TRUE, 0, NUM_PIXELS * sizeof(float), ndvi_result, 0, NULL, NULL);

    Mat ndvi_image(HEIGHT, WIDTH, CV_32F, ndvi_result);
    Mat ndvi_normalized;
    ndvi_image = (ndvi_image + 1) / 2.0 * 255.0;
    ndvi_image.convertTo(ndvi_normalized, CV_8UC1);
    imwrite(output_filepath, ndvi_normalized);

    free(nir_image);
    free(red_image);
    free(ndvi_result);
}

int main() {
    const std::string nir_folder = "/home/karim/Desktop/archive(2)/Multispectral-images/NIR";
    const std::string red_folder = "/home/karim/Desktop/archive(2)/Multispectral-images/RED";
    const std::string output_folder = "/home/karim/Desktop/parallelism/vegetation_index/NDVI2";
    
    DIR* dir = opendir(nir_folder.c_str());
    struct dirent* entry;
    std::string first_nir, first_red;
    
    if (dir) {
        while ((entry = readdir(dir)) != NULL) {
            if (strstr(entry->d_name, ".tif")) {
                first_nir = nir_folder + "/" + entry->d_name;
                first_red = red_folder + "/" + entry->d_name;
                break; 
            }
        }
        closedir(dir);
    }

    if (first_nir.empty() || first_red.empty()) {
        printf("No valid images found.\n");
        return 1;
    }

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_queue_properties properties[] = {0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, properties, NULL);
    
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "calculate_ndvi", NULL);

    cl_mem bufferNIR = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM_PIXELS * sizeof(float), NULL, NULL);
    cl_mem bufferRED = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM_PIXELS * sizeof(float), NULL, NULL);
    cl_mem bufferNDVI = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NUM_PIXELS * sizeof(float), NULL, NULL);

    std::string output_filepath = output_folder + "/ndvi_result.tif";
auto start = std::chrono::high_resolution_clock::now();    
process_image(first_nir, first_red, output_filepath, queue, kernel, bufferNIR, bufferRED, bufferNDVI);
auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration = end - start;

// Print the time taken
std::cout << "Time taken to process the first NDVI image: " << duration.count() << " seconds" << std::endl;

    clReleaseMemObject(bufferNIR);
    clReleaseMemObject(bufferRED);
    clReleaseMemObject(bufferNDVI);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
}
