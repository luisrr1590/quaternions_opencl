float4 QuaternionProd (float4 q1, float4 q2)
{ 
	float4 prod;
	
	prod.x = (q1.w * q2.x) + (q1.z * q2.y) - (q1.y * q2.z) + (q1.x * q2.w);
	prod.y = (q1.w * q2.y) - (q1.z * q2.x) + (q1.y * q2.w) + (q1.x * q2.z);
	prod.z = (q1.w * q2.z) + (q1.z * q2.w) + (q1.y * q2.x) - (q1.x * q2.y);
	prod.w = (q1.w * q2.w) - (q1.z * q2.z) - (q1.y * q2.y) - (q1.x * q2.x);

	return prod;
}

__kernel void quaternion_conv_buffers(__global const float4 * inputImage, __global const float4 * leftMask, __global const float4 * rightMask, __global float4 * outputImage)
{
	const int x     = get_global_id(0);
    const int y     = get_global_id(1);
    const int width = get_global_size(0);
	const int height = get_global_size(1);

	const int id = y * width + x;

	float4 hL, hR, pixel, outputPixel;

	if (get_global_id(0) >= 1 && get_global_id(1) >= 1 && get_global_id(0) <= width - 2 && get_global_id(1) <= height - 2)
	{
		//printf("\nEvaluando pixel: %d, %d ; id = %d\n\n", x,y,id);
		
		#pragma unroll
		for(int a = -1; a < 2; a++) 
		{
			#pragma unroll
			for(int b = -1; b < 2; b++) 
			{	
				hL = leftMask[(a+1) * 3 + (b+1)];
				hR = rightMask[(a+1) * 3 + (b+1)];					
				int pos = id + (a)*width+(b);
				pixel = inputImage[pos];
				//printf("id + (a)*width + (b) = %d + (%d)*%d + (%d) = %d\n", a,width,b,id, id + (a)*width+(b));
				//printf("pixel: r = %f\n", pixel.z);
				float4 leftProd = QuaternionProd(hL, pixel);
				float4 rightProd = QuaternionProd(leftProd, hR);
				outputPixel += rightProd; 
				outputPixel.w = 0.0f;
				//barrier(CLK_LOCAL_MEM_FENCE);
			}
		}

		outputImage[id] = outputPixel;
	}
}
