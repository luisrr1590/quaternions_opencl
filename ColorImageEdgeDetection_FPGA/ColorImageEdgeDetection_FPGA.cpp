/*****************************************************************************
 * Copyright (c) 2013-2016 Intel Corporation
 * All rights reserved.
 *
 * WARRANTY DISCLAIMER
 *
 * THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
 * MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Intel Corporation is the author of the Materials, and requests that all
 * problem reports or change requests be submitted to it directly
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <tchar.h>
#include <memory.h>
#include <vector>

#include "CL\cl.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

 //for perf. counters
#include <Windows.h>

using namespace std;
using namespace cv;


//for perf. counters
#include <Windows.h>


// Macros for OpenCL versions
#define OPENCL_VERSION_1_2  1.2f
#define OPENCL_VERSION_2_0  2.0f

Mat saturationQ(Mat RGBimage)
{
	Mat result = Mat(RGBimage.rows, RGBimage.cols, CV_32FC1);

	std::vector<cv::Mat> planes(3);
	cv::split(RGBimage, planes);
	Mat maxMatrix = cv::Mat(cv::max(planes[2], cv::max(planes[1], planes[0])));
	Mat minMatrix = cv::Mat(cv::min(planes[2], cv::min(planes[1], planes[0])));

	Mat delta = maxMatrix - minMatrix;

	for (int i = 0; i < RGBimage.rows; i++)
		for (int j = 0; j < RGBimage.cols; j++)
		{
			if (maxMatrix.at<float>(i, j) == 0)
				result.at<float>(i, j) = 0;
			else
				result.at<float>(i, j) = delta.at<float>(i, j) / maxMatrix.at<float>(i, j);
		}

	return result;
}


/* This function helps to create informative messages in
 * case when OpenCL errors occur. It returns a string
 * representation for an OpenCL error code.
 * (E.g. "CL_DEVICE_NOT_FOUND" instead of just -1.)
 */
const char* TranslateOpenCLError(cl_int errorCode)
{
    switch(errorCode)
    {
    case CL_SUCCESS:                            return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          //-13
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   //-14
    case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               //-15
    case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  //-16
    case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  //-17
    case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               //-18
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         //-19
    case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           //-63
    case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   //-64
    case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           //-65
    case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           //-66
    case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             //-67
    case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     //-68
//    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
//    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70    

    default:
        return "UNKNOWN ERROR CODE";
    }
}


/* Convenient container for all OpenCL specific objects used in the sample
 *
 * It consists of two parts:
 *   - regular OpenCL objects which are used in almost each normal OpenCL applications
 *   - several OpenCL objects that are specific for this particular sample
 *
 * You collect all these objects in one structure for utility purposes
 * only, there is no OpenCL specific here: just to avoid global variables
 * and make passing all these arguments in functions easier.
 */
struct ocl_args_d_t
{
    ocl_args_d_t();
    ~ocl_args_d_t();

    // Regular OpenCL objects:
    cl_context       context;           // hold the context handler
    cl_device_id     device;            // hold the selected device handler
    cl_command_queue commandQueue;      // hold the commands-queue handler
    cl_program       program;           // hold the program handler
    cl_kernel        kernel;            // hold the kernel handler
    float            platformVersion;   // hold the OpenCL platform version (default 1.2)
    float            deviceVersion;     // hold the OpenCL device version (default. 1.2)
    float            compilerVersion;   // hold the device OpenCL C version (default. 1.2)
    
    // Objects that are specific for algorithm implemented in this sample
    cl_mem           inputImage;              // hold first source buffer
    cl_mem           leftMask;              // hold leftMask buffer
    cl_mem           rightMask;            // hold rightMask buffer
	cl_mem           outputImage;            // hold destination buffer
};

ocl_args_d_t::ocl_args_d_t():
        context(NULL),
        device(NULL),
        commandQueue(NULL),
        program(NULL),
        kernel(NULL),
        platformVersion(OPENCL_VERSION_1_2),
        deviceVersion(OPENCL_VERSION_1_2),
        compilerVersion(OPENCL_VERSION_1_2),
        inputImage(NULL),
		leftMask(NULL),
		rightMask(NULL),
        outputImage(NULL)
{
}

/*
 * destructor - called only once
 * Release all OpenCL objects
 * This is a regular sequence of calls to deallocate all created OpenCL resources in bootstrapOpenCL.
 *
 * You may want to call these deallocation procedures in the middle of your application execution
 * (not at the end) if you don't further need OpenCL runtime.
 * You may want to do that in order to free some memory, for example,
 * or recreate OpenCL objects with different parameters.
 *
 */
ocl_args_d_t::~ocl_args_d_t()
{
    cl_int err = CL_SUCCESS;

    if (kernel)
    {
        err = clReleaseKernel(kernel);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
        }
    }
    if (program)
    {
        err = clReleaseProgram(program);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseProgram returned '%s'.\n", TranslateOpenCLError(err));
        }
    }
    if (inputImage)
    {
        err = clReleaseMemObject(inputImage);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
        }
    }
    if (leftMask)
    {
        err = clReleaseMemObject(leftMask);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
        }
    }
    if (rightMask)
    {
        err = clReleaseMemObject(rightMask);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
        }
    }
	if (outputImage)
	{
		err = clReleaseMemObject(outputImage);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
    if (commandQueue)
    {
        err = clReleaseCommandQueue(commandQueue);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseCommandQueue returned '%s'.\n", TranslateOpenCLError(err));
        }
    }
    if (device)
    {
        err = clReleaseDevice(device);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseDevice returned '%s'.\n", TranslateOpenCLError(err));
        }
    }
    if (context)
    {
        err = clReleaseContext(context);
        if (CL_SUCCESS != err)
        {
            LogError("Error: clReleaseContext returned '%s'.\n", TranslateOpenCLError(err));
        }
    }

    /*
     * Note there is no procedure to deallocate platform 
     * because it was not created at the startup,
     * but just queried from OpenCL runtime.
     */
}


/*
 * Check whether an OpenCL platform is the required platform
 * (based on the platform's name)
 */
bool CheckPreferredPlatformMatch(cl_platform_id platform, const char* preferredPlatform)
{
    size_t stringLength = 0;
    cl_int err = CL_SUCCESS;
    bool match = false;

    // In order to read the platform's name, we first read the platform's name string length (param_value is NULL).
    // The value returned in stringLength
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &stringLength);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME length returned '%s'.\n", TranslateOpenCLError(err));
        return false;
    }

    // Now, that we know the platform's name string length, we can allocate enough space before read it
    std::vector<char> platformName(stringLength);

    // Read the platform's name string
    // The read value returned in platformName
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, stringLength, &platformName[0], NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetplatform_ids() to get CL_PLATFORM_NAME returned %s.\n", TranslateOpenCLError(err));
        return false;
    }
    
	for (std::vector<char>::iterator i = platformName.begin(); i != platformName.end(); ++i)
	{
		printf("%c", *i);
	}

    // Now check if the platform's name is the required one
    if (strstr(&platformName[0], preferredPlatform) != 0)
    {
        // The checked platform is the one we're looking for
        match = true;
		printf(" Match: TRUE\n");
	}
	else
		printf(" Match: FALSE\n");

    return match;
}

/*
 * Find and return the preferred OpenCL platform
 * In case that preferredPlatform is NULL, the ID of the first discovered platform will be returned
 */
cl_platform_id FindOpenCLPlatform(const char* preferredPlatform, cl_device_type deviceType)
{
    cl_uint numPlatforms = 0;
    cl_int err = CL_SUCCESS;

    // Get (in numPlatforms) the number of OpenCL platforms available
    // No platform ID will be return, since platforms is NULL
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetplatform_ids() to get num platforms returned %s.\n", TranslateOpenCLError(err));
        return NULL;
    }
    LogInfo("Number of available platforms: %u\n\n", numPlatforms);

    if (0 == numPlatforms)
    {
        LogError("Error: No platforms found!\n");
        return NULL;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);

    // Now, obtains a list of numPlatforms OpenCL platforms available
    // The list of platforms available will be returned in platforms
    err = clGetPlatformIDs(numPlatforms, &platforms[0], NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetplatform_ids() to get platforms returned %s.\n", TranslateOpenCLError(err));
        return NULL;
    }

    // Check if one of the available platform matches the preferred requirements
    for (cl_uint i = 0; i < numPlatforms; i++)
    {
        bool match = true;
        cl_uint numDevices = 0;

        // If the preferredPlatform is not NULL then check if platforms[i] is the required one
        // Otherwise, continue the check with platforms[i]
        if ((NULL != preferredPlatform) && (strlen(preferredPlatform) > 0))
        {
            // In case we're looking for a specific platform
            match = CheckPreferredPlatformMatch(platforms[i], preferredPlatform);
        }

        // match is true if the platform's name is the required one or don't care (NULL)
        if (match)
        {
            // Obtains the number of deviceType devices available on platform
            // When the function failed we expect numDevices to be zero.
            // We ignore the function return value since a non-zero error code
            // could happen if this platform doesn't support the specified device type.
            err = clGetDeviceIDs(platforms[i], deviceType, 0, NULL, &numDevices);
            if (CL_SUCCESS != err)
            {
                LogError("clGetDeviceIDs() returned %s.\n", TranslateOpenCLError(err));
            }

            if (0 != numDevices)
            {
                // There is at list one device that answer the requirements
                return platforms[i];
            }
        }
    }

    return NULL;
}


/*
 * This function read the OpenCL platdorm and device versions
 * (using clGetxxxInfo API) and stores it in the ocl structure.
 * Later it will enable us to support both OpenCL 1.2 and 2.0 platforms and devices
 * in the same program.
 */
int GetPlatformAndDeviceVersion (cl_platform_id platformId, ocl_args_d_t *ocl)
{
    cl_int err = CL_SUCCESS;

    // Read the platform's version string length (param_value is NULL).
    // The value returned in stringLength
    size_t stringLength = 0;
    err = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, 0, NULL, &stringLength);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetPlatformInfo() to get CL_PLATFORM_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
        return err;
    }

    // Now, that we know the platform's version string length, we can allocate enough space before read it
    std::vector<char> platformVersion(stringLength);

    // Read the platform's version string
    // The read value returned in platformVersion
    err = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, stringLength, &platformVersion[0], NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetplatform_ids() to get CL_PLATFORM_VERSION returned %s.\n", TranslateOpenCLError(err));
        return err;
    }

    if (strstr(&platformVersion[0], "OpenCL 2.0") != NULL)
    {
        ocl->platformVersion = OPENCL_VERSION_2_0;
    }

    // Read the device's version string length (param_value is NULL).
    err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, 0, NULL, &stringLength);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
        return err;
    }

    // Now, that we know the device's version string length, we can allocate enough space before read it
    std::vector<char> deviceVersion(stringLength);

    // Read the device's version string
    // The read value returned in deviceVersion
    err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, stringLength, &deviceVersion[0], NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION returned %s.\n", TranslateOpenCLError(err));
        return err;
    }

    if (strstr(&deviceVersion[0], "OpenCL 2.0") != NULL)
    {
        ocl->deviceVersion = OPENCL_VERSION_2_0;
    }

    // Read the device's OpenCL C version string length (param_value is NULL).
    err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &stringLength);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
        return err;
    }

    // Now, that we know the device's OpenCL C version string length, we can allocate enough space before read it
    std::vector<char> compilerVersion(stringLength);

    // Read the device's OpenCL C version string
    // The read value returned in compilerVersion
    err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, stringLength, &compilerVersion[0], NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION returned %s.\n", TranslateOpenCLError(err));
        return err;
    }

    else if (strstr(&compilerVersion[0], "OpenCL C 2.0") != NULL)
    {
        ocl->compilerVersion = OPENCL_VERSION_2_0;
    }

    return err;
}

/*
 * This function picks/creates necessary OpenCL objects which are needed.
 * The objects are:
 * OpenCL platform, device, context, and command queue.
 *
 * All these steps are needed to be performed once in a regular OpenCL application.
 * This happens before actual compute kernels calls are performed.
 *
 * For convenience, in this application you store all those basic OpenCL objects in structure ocl_args_d_t,
 * so this function populates fields of this structure, which is passed as parameter ocl.
 * Please, consider reviewing the fields before going further.
 * The structure definition is right in the beginning of this file.
 */
int SetupOpenCL(ocl_args_d_t *ocl, cl_device_type deviceType, const char* preferredPlatform)
{
    // The following variable stores return codes for all OpenCL calls.
    cl_int err = CL_SUCCESS;

    // Query for all available OpenCL platforms on the system
    // Here you enumerate all platforms and pick one which name has preferredPlatform as a sub-string
    cl_platform_id platformId = FindOpenCLPlatform(preferredPlatform, deviceType);
    if (NULL == platformId)
    {
        LogError("Error: Failed to find OpenCL platform.\n");
        return CL_INVALID_VALUE;
    }

    // Create context with device of specified type.
    // Required device type is passed as function argument deviceType.
    // So you may use this function to create context for any CPU or GPU OpenCL device.
    // The creation is synchronized (pfn_notify is NULL) and NULL user_data
    cl_context_properties contextProperties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platformId, 0};
    ocl->context = clCreateContextFromType(contextProperties, deviceType, NULL, NULL, &err);
    if ((CL_SUCCESS != err) || (NULL == ocl->context))
    {
        LogError("Couldn't create a context, clCreateContextFromType() returned '%s'.\n", TranslateOpenCLError(err));
        return err;
    }

    // Query for OpenCL device which was used for context creation
    err = clGetContextInfo(ocl->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &ocl->device, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clGetContextInfo() to get list of devices returned %s.\n", TranslateOpenCLError(err));
        return err;
    }

    // Read the OpenCL platform's version and the device OpenCL and OpenCL C versions
    GetPlatformAndDeviceVersion(platformId, ocl);

    // Create command queue.
    // OpenCL kernels are enqueued for execution to a particular device through special objects called command queues.
    // Command queue guarantees some ordering between calls and other OpenCL commands.
    // Here you create a simple in-order OpenCL command queue that doesn't allow execution of two kernels in parallel on a target device.
#ifdef CL_VERSION_2_0
    if (OPENCL_VERSION_2_0 == ocl->deviceVersion)
    {
        const cl_command_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        ocl->commandQueue = clCreateCommandQueueWithProperties(ocl->context, ocl->device, properties, &err);
    } 
    else {
        // default behavior: OpenCL 1.2
        cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
        ocl->commandQueue = clCreateCommandQueue(ocl->context, ocl->device, properties, &err);
    } 
#else
    // default behavior: OpenCL 1.2
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
    ocl->commandQueue = clCreateCommandQueue(ocl->context, ocl->device, properties, &err);
#endif
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateCommandQueue() returned %s.\n", TranslateOpenCLError(err));
        return err;
    }

    return CL_SUCCESS;
}


/* 
 * Create and build OpenCL program from its source code
 */
int CreateAndBuildProgram(ocl_args_d_t *ocl)
{
    cl_int err = CL_SUCCESS;

    // Upload the OpenCL C source code from the input file to source
    // The size of the C program is returned in sourceSize
    char* source = NULL;
    size_t src_size = 0;
    err = ReadSourceFromFile("quaternion_conv_buffers.cl", &source, &src_size);
    if (CL_SUCCESS != err)
    {
        LogError("Error: ReadSourceFromFile returned %s.\n", TranslateOpenCLError(err));
        goto Finish;
    }

    // And now after you obtained a regular C string call clCreateProgramWithSource to create OpenCL program object.
    ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**)&source, &src_size, &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateProgramWithSource returned %s.\n", TranslateOpenCLError(err));
        goto Finish;
    }

    // Build the program
    // During creation a program is not built. You need to explicitly call build function.
    // Here you just use create-build sequence,
    // but there are also other possibilities when program consist of several parts,
    // some of which are libraries, and you may want to consider using clCompileProgram and clLinkProgram as
    // alternatives.
    err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));

        // In case of error print the build log to the standard output
        // First check the size of the log
        // Then allocate the memory and obtain the log from the program
        if (err == CL_BUILD_PROGRAM_FAILURE)
        {
            size_t log_size = 0;
            clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

            std::vector<char> build_log(log_size);
            clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], NULL);

            LogError("Error happened during the build of OpenCL program.\nBuild log:%s", &build_log[0]);
        }
    }

Finish:
    if (source)
    {
        delete[] source;
        source = NULL;
    }

    return err;
}


/*
 * Create OpenCL buffers from host memory
 * These buffers will be used later by the OpenCL kernel
 */
int CreateBufferArguments(ocl_args_d_t *ocl, Mat inputImage, Mat leftMask, Mat rightMask, cl_uint arrayWidth, cl_uint arrayHeight, cl_float4 * data)
{
    cl_int err = CL_SUCCESS;

    // Create new OpenCL buffer objects
    // As these buffer are used only for read by the kernel, you are recommended to create it with flag CL_MEM_READ_ONLY.
    // Always set minimal read/write flags for buffers, it may lead to better performance because it allows runtime
    // to better organize data copying.
    // You use CL_MEM_COPY_HOST_PTR here, because the buffers should be populated with bytes at inputA and inputB.

    ocl->inputImage = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float4) * arrayWidth * arrayHeight, inputImage.data, &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateBuffer for inputImage returned %s\n", TranslateOpenCLError(err));
        return err;
    }

	ocl->leftMask = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float4) * 3 * 3, leftMask.data, &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateBuffer for leftMask returned %s\n", TranslateOpenCLError(err));
        return err;
    }

	ocl->rightMask = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float4) * 3 * 3, rightMask.data, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateBuffer for rightMask returned %s\n", TranslateOpenCLError(err));
		return err;
	}

	ocl->outputImage = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float4) * arrayWidth * arrayHeight, data, &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateBuffer for outputImage returned %s\n", TranslateOpenCLError(err));
        return err;
    }


    return CL_SUCCESS;
}


/*
 * Set kernel arguments
 */
cl_uint SetKernelArguments(ocl_args_d_t *ocl)
{
    cl_int err = CL_SUCCESS;

    err  =  clSetKernelArg(ocl->kernel, 0, sizeof(cl_mem), (void *)&ocl->inputImage);
    if (CL_SUCCESS != err)
    {
        LogError("error: Failed to set argument srcA, returned %s\n", TranslateOpenCLError(err));
        return err;
    }

    err  = clSetKernelArg(ocl->kernel, 1, sizeof(cl_mem), (void *)&ocl->leftMask);
    if (CL_SUCCESS != err)
    {
        LogError("Error: Failed to set argument srcB, returned %s\n", TranslateOpenCLError(err));
        return err;
    }

	err = clSetKernelArg(ocl->kernel, 2, sizeof(cl_mem), (void *)&ocl->rightMask);
	if (CL_SUCCESS != err)
	{
		LogError("Error: Failed to set argument srcB, returned %s\n", TranslateOpenCLError(err));
		return err;
	}

    err  = clSetKernelArg(ocl->kernel, 3, sizeof(cl_mem), (void *)&ocl->outputImage);
    if (CL_SUCCESS != err)
    {
        LogError("Error: Failed to set argument dstMem, returned %s\n", TranslateOpenCLError(err));
        return err;
    }

    return err;
}


/*
 * Execute the kernel
 */
cl_uint ExecuteKernel(ocl_args_d_t *ocl, cl_uint width, cl_uint height)
{
    cl_int err = CL_SUCCESS;

    // Define global iteration space for clEnqueueNDRangeKernel.
    size_t globalWorkSize[2] = {width, height};


    // execute kernel
    err = clEnqueueNDRangeKernel(ocl->commandQueue, ocl->kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: Failed to run kernel, return %s\n", TranslateOpenCLError(err));
        return err;
    }

    // Wait until the queued kernel is completed by the device
    err = clFinish(ocl->commandQueue);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clFinish return %s\n", TranslateOpenCLError(err));
        return err;
    }

    return CL_SUCCESS;
}


/*
 * main execution routine
 * Basically it consists of three parts:
 *   - generating the inputs
 *   - running OpenCL kernel
 *   - reading results of processing
 */
int _tmain(int argc, TCHAR* argv[])
{
	Mat imgRGBA, image_f;
	cvtColor(imread("../lena.tiff"), imgRGBA, COLOR_RGB2RGBA);

	if (!imgRGBA.empty())
	{
		namedWindow("RGB Orig Image", WINDOW_AUTOSIZE);
		imshow("RGB Orig Image", imgRGBA);
		printf("Amount of Channels Orig: %d \n", imgRGBA.channels());
		cout << imgRGBA.type() << " == " << CV_8UC4 << endl;
		imgRGBA.convertTo(image_f, CV_32F, 1.0f / 255, 0);
		cout << image_f.type() << " == " << CV_32FC4 << endl;
	}
	else
	{
		printf("Image couldnt be loaded");
		return 0;
	}
	
	cl_int err;
    ocl_args_d_t ocl;
    cl_device_type deviceType = CL_DEVICE_TYPE_CPU;

    LARGE_INTEGER perfFrequency;
    LARGE_INTEGER performanceCountNDRangeStart;
    LARGE_INTEGER performanceCountNDRangeStop;

    cl_uint arrayWidth  = imgRGBA.cols;
    cl_uint arrayHeight = imgRGBA.rows;

    //initialize Open CL objects (context, queue, etc.)
    if (CL_SUCCESS != SetupOpenCL(&ocl, deviceType, "Intel(R) CPU Runtime for OpenCL(TM) Applications"))
    {
		waitKey(0);
		return -1;
    }

	float left[3][3][4] =
	{
	{ { 0.1667f, 0.1667f, 0.1667, 0.2887f },{ 0.1667f, 0.1667f, 0.1667, 0.2887f },{ 0.1667f, 0.1667f, 0.1667, 0.2887f } },
	{ { 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 0.0f, 0.0f } },
	{ { -0.1667f, -0.1667f, -0.1667, 0.2887f },{ -0.1667f, -0.1667f, -0.1667, 0.2887f },{ -0.1667f, -0.1667f, -0.1667, 0.2887f } }
	};

	float right[3][3][4] =
	{
		{{-0.1667f, -0.1667f, -0.1667, 0.2887f}, { -0.1667f, -0.1667f, -0.1667, 0.2887f }, { -0.1667f, -0.1667f, -0.1667, 0.2887f }},
		{{ 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }},
		{{ 0.1667f, 0.1667f, 0.1667, 0.2887f }, { 0.1667f, 0.1667f, 0.1667, 0.2887f }, { 0.1667f, 0.1667f, 0.1667, 0.2887f } }
	};

	Mat leftMask = Mat(3, 3, CV_32FC4, &left);
	Mat rightMask = Mat(3, 3, CV_32FC4, &right);

    // Create OpenCL buffers from host memory
    // These buffers will be used later by the OpenCL kernel
	cl_float4 * data = new cl_float4[arrayWidth*arrayHeight];
	if (CL_SUCCESS != CreateBufferArguments(&ocl, image_f, leftMask, rightMask, arrayWidth, arrayHeight, data))
    {
		waitKey(0);
		return -1;
    }

     // Create and build the OpenCL program
    if (CL_SUCCESS != CreateAndBuildProgram(&ocl))
    {
		waitKey(0);
		return -1;
    }

    // Program consists of kernels.
    // Each kernel can be called (enqueued) from the host part of OpenCL application.
    // To call the kernel, you need to create it from existing program.
    ocl.kernel = clCreateKernel(ocl.program, "quaternion_conv_buffers", &err);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
		waitKey(0);
		return -1;
    }

    // Passing arguments into OpenCL kernel.
    if (CL_SUCCESS != SetKernelArguments(&ocl))
    {
		waitKey(0);
		return -1;
    }

    // Regularly you wish to use OpenCL in your application to achieve greater performance results
    // that are hard to achieve in other ways.
    // To understand those performance benefits you may want to measure time your application spent in OpenCL kernel execution.
    // The recommended way to obtain this time is to measure interval between two moments:
    //   - just before clEnqueueNDRangeKernel is called, and
    //   - just after clFinish is called
    // clFinish is necessary to measure entire time spending in the kernel, measuring just clEnqueueNDRangeKernel is not enough,
    // because this call doesn't guarantees that kernel is finished.
    // clEnqueueNDRangeKernel is just enqueue new command in OpenCL command queue and doesn't wait until it ends.
    // clFinish waits until all commands in command queue are finished, that suits your need to measure time.
    bool queueProfilingEnable = true;
    if (queueProfilingEnable)
        QueryPerformanceCounter(&performanceCountNDRangeStart);
    // Execute (enqueue) the kernel
    if (CL_SUCCESS != ExecuteKernel(&ocl, arrayWidth, arrayHeight))
    {
		waitKey(0);
		return -1;
    }
    if (queueProfilingEnable)
        QueryPerformanceCounter(&performanceCountNDRangeStop);

	// retrieve performance counter frequency
    if (queueProfilingEnable)
    {
        QueryPerformanceFrequency(&perfFrequency);
        LogInfo("NDRange performance counter time %f ms.\n",
            1000.0f*(float)(performanceCountNDRangeStop.QuadPart - performanceCountNDRangeStart.QuadPart) / (float)perfFrequency.QuadPart);
    }

	Mat outputImageRGBA(imgRGBA.size(), CV_32FC4, data);

	if (!outputImageRGBA.empty())
	{
		namedWindow("RGB Proc Image", WINDOW_AUTOSIZE);
		imshow("RGB Proc Image", outputImageRGBA);
		printf("Amount of Channels Proc: %d \n", outputImageRGBA.channels());
	}

	Mat RGB, saturation;
	cvtColor(outputImageRGBA, RGB, COLOR_RGBA2RGB);

	saturation = saturationQ(RGB);
	if (!saturation.empty())
	{
		namedWindow("Saturation", WINDOW_AUTOSIZE);
		imshow("Saturation", saturation);
	}

	waitKey(0);
	return 0;
}

