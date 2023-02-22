#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

#ifndef __FLT_MAX__
#define __FLT_MAX__ 3.40282347e+38
#endif

struct Point {
	float x, y;     // coordinates
	int cluster;     // no default cluster
	float minDist;  // default infinite dist to nearest cluster
};

float dist(float px, float py, float cx, float cy)
{
    float distance = ((px - cx) * (px - cx)) + ((py - cy) * (py - cy));
    return distance;
}

__kernel void assignCluster(__global struct Point* points, __global struct Point* centroids, __global float* sumX, __global float* sumY, __global int* nPoints, int k) {
	size_t g_index = get_global_id(0);
	size_t l_index = get_local_id(0);

	struct Point p = points[g_index];

	for (int i = 0; i < k; i++) {

		float distance = dist(p.x, p.y, centroids[i].x, centroids[i].y);
		if (distance < p.minDist) {
			p.minDist = distance;
			p.cluster = i;
		}
		points[g_index] = p;
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 0; i < k; i++) {
		int clusterId = p.cluster;
		nPoints[clusterId]++;
		sumX[clusterId]+= p.x;
		sumY[clusterId]+= p.y;

		p.minDist = FLT_MAX;
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void generateNewCentroids(__global struct Point* centroids, __global float* sumX, __global float* sumY, __global int* nPoints) {
	size_t index = get_global_id(0);

	float x = sumX[index] / nPoints[index];
	float y = sumY[index] / nPoints[index];

	barrier(CLK_GLOBAL_MEM_FENCE);

	sumX[index] = 0.0;
	sumY[index] = 0.0;
	nPoints[index] = 0;

	barrier(CLK_GLOBAL_MEM_FENCE);

	centroids[index].x = x;
	centroids[index].y = y;

	barrier(CLK_GLOBAL_MEM_FENCE);
}