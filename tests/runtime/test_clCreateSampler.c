/* Tests clCreateSampler

   Copyright (c) 2017 James Price / University of Bristol

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/
#include <CL/cl.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "poclu.h"

char kernelSourceCode[] =
"kernel void test_sampler(read_only  image2d_t input,         \n"
"                         write_only image2d_t output,        \n"
"                                    sampler_t sampler)       \n"
"{                                                            \n"
"  int x = get_global_id(0);                                  \n"
"  int y = get_global_id(1);                                  \n"
"  float4 pixel = read_imagef(input, sampler, (int2)(x,y));   \n"
"  write_imagef(output, (int2)(x,y), pixel);                  \n"
"}\n";

int main()
{
  const int N = 64;
  size_t global_work_size[2] = { N, N };
  cl_int err;
  cl_platform_id platforms[1];
  cl_uint nplatforms;
  cl_device_id devices[1]; // + 1 for duplicate test
  cl_uint num_devices;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem input = NULL;
  cl_mem output = NULL;
  cl_sampler sampler = NULL;
  cl_command_queue queue = NULL;

  err = clGetPlatformIDs(1, platforms, &nplatforms);
  CHECK_OPENCL_ERROR_IN("clGetPlatformIDs");
  if (!nplatforms)
    return EXIT_FAILURE;

  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1,
                       devices, &num_devices);
  CHECK_OPENCL_ERROR_IN("clGetDeviceIDs");

  cl_context context = clCreateContext(NULL, num_devices, devices, NULL,
                                       NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateContext");

  err = clGetContextInfo(context, CL_CONTEXT_DEVICES,
                         sizeof(cl_device_id), devices, NULL);
  CHECK_OPENCL_ERROR_IN("clGetContextInfo");

  queue = clCreateCommandQueue(context, devices[0], 0, &err);
  CHECK_OPENCL_ERROR_IN("clCreateCommandQueue");
  TEST_ASSERT(queue);

  // Host image data
  uint8_t *input_data = malloc(N*N*4);
  uint8_t *output_data = malloc(N*N*4);
  for (int i = 0; i < N*N*4; i++)
  {
    input_data[i] = i % 256 / 2;
    output_data[i] = 0;
  }

  cl_image_format format;
  format.image_channel_order = CL_RGBA;
  format.image_channel_data_type = CL_UNORM_INT8;

  cl_image_desc desc = {0};
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = N;
  desc.image_height = N;

  input = clCreateImage(context,
                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        &format, &desc, input_data, &err);
  CHECK_OPENCL_ERROR_IN("clCreateImage");
  TEST_ASSERT(input);

  output = clCreateImage(context,
                         CL_MEM_WRITE_ONLY,
                         &format, &desc, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateImage");
  TEST_ASSERT(input);


  sampler = clCreateSampler(context,
                            CL_FALSE,
                            CL_ADDRESS_CLAMP_TO_EDGE,
                            CL_FILTER_NEAREST,
                            &err);
  CHECK_OPENCL_ERROR_IN("clCreateSampler");
  TEST_ASSERT(sampler);


  size_t kernel_size = strlen (kernelSourceCode);
  char* kernel_buffer = kernelSourceCode;

  program = clCreateProgramWithSource (context, 1,
                                       (const char**)&kernel_buffer,
                                       &kernel_size, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

  err = clBuildProgram (program, num_devices, devices, NULL, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clBuildProgram");

  kernel = clCreateKernel (program, "test_sampler", NULL);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");
  TEST_ASSERT(kernel);

  err = clSetKernelArg (kernel, 0, sizeof (cl_mem), &input);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg");

  err = clSetKernelArg (kernel, 1, sizeof (cl_mem), &output);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg");

  err = clSetKernelArg (kernel, 2, sizeof (cl_sampler), &sampler);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg");


  err = clEnqueueNDRangeKernel (queue, kernel, 2, NULL, global_work_size,
                                NULL, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel");

  clFinish(queue);


  size_t origin[] = {0, 0, 0};
  size_t region[] = {N, N, 1};
  err = clEnqueueReadImage(queue, output, CL_TRUE, origin, region, 0, 0,
                           output_data, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadImage");


  for (int y = 0; y < N; y++)
  {
    for (int x = 0; x < N; x++)
    {
      int xleft = x ? x - 1 : 0;
      for (int c = 0; c < 4; c++)
      {
        int i = (x + y*N)*4 + c;
        int ref = input_data[i];
        if (output_data[i] != ref)
        {
          fprintf(stderr, "%2d,%2d,%2d: %d != %d\n", x, y, c, output_data[i], ref);
          return EXIT_FAILURE;
        }
      }

    }
  }

  printf("OK\n");
  return EXIT_SUCCESS;
}
