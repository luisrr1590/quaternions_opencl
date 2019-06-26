__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

float4 QuaternionProd (float4 q1, float4 q2)
{ 
	float4 prod;
	
	prod.x = (q1.w * q2.x) + (q1.z * q2.y) - (q1.y * q2.z) + (q1.x * q2.w);
	prod.y = (q1.w * q2.y) - (q1.z * q2.x) + (q1.y * q2.w) + (q1.x * q2.z);
	prod.z = (q1.w * q2.z) + (q1.z * q2.w) + (q1.y * q2.x) - (q1.x * q2.y);
	prod.w = (q1.w * q2.w) - (q1.z * q2.z) - (q1.y * q2.y) - (q1.x * q2.x);

	return prod;
}

__kernel void quaternion_conv_images( __read_only image2d_t image, __read_only image2d_t leftMask, __read_only image2d_t rightMask, __write_only image2d_t processedImage) 
{	
	const int2 pos = {get_global_id(0), get_global_id(1)};
	float4 hL, hR, pixel, outputPixel;
	
	#pragma unroll
	for(int a = 0; a < 3; a++) 
	{
		#pragma unroll
		for(int b = 0; b < 3; b++) 
		{
			hL = read_imagef(leftMask, sampler, (int2)(a,b));
			hR = read_imagef(rightMask, sampler, (int2)(a,b));
			pixel = read_imagef(image, sampler, pos - (int2)(a+1,b+1));
			float4 leftProd = QuaternionProd(hL, pixel);
			float4 rightProd = QuaternionProd(leftProd, hR);
			outputPixel += rightProd; 
			outputPixel.w = 0.0f;
		    //barrier(CLK_LOCAL_MEM_FENCE);
        }
	}

	write_imagef(processedImage, pos, outputPixel);	
}