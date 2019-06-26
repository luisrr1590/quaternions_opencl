# quaternions_opencl
A color image processing OpenCL implementation based on the quaternions convolution algorithm by S.J. Sangwine.

The masks are based on the Prewitt horizontal edge mask.


There are two implemetations:

The first one is based on images and samplers for an easy kernel implementation.

The second one is based on buffers, this is usable with the Intel OpenCL SDK for FPGAs given that this SDK does not support images and samplers.
