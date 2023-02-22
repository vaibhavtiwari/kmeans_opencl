// includes
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <string>
#include <numeric>

//include Libraries for PCA
#include "OpenCL/qbMatrix.h"
#include "OpenCL/qbVector.h"
#include "OpenCL/qbPCA.h"

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
template <typename T>
std::vector< std::vector<T>> vect_Transpose(std::vector<std::vector<T>>& input_Vector)
{
	if (input_Vector.size() > 0)
	{
		std::vector<std::vector<T> > out_Vector(input_Vector[0].size(), std::vector<T>(input_Vector.size()));
		for (int i = 0; i < input_Vector.size(); i++)
		{
			for (int j = 0; j < input_Vector[i].size(); j++)
			{
				out_Vector[j][i] = input_Vector[i][j];
			}
		}
		return out_Vector;
	}
	return input_Vector;
}

std::vector<float> findMinMax(std::vector<float> arr) {
	float max = arr[0];
	float min = arr[0];
	for (int i = 1; i < arr.size(); i++) {
		if (arr[i] > max) {
			max = arr[i];
		}
		if (arr[i] < min) {
			min = arr[i];
		}
	}
	std::vector<float> minmax;
	minmax.push_back(min);
	minmax.push_back(max);

	return minmax;
}

struct Point {
	float x, y;     // coordinates
	int cluster;     // no default cluster
	float minDist;  // default infinite dist to nearest cluster

	Point() :
		x(0.0),
		y(0.0),
		cluster(-1),
		minDist(FLT_MAX) {}

	Point(float x, float y) :
		x(x),
		y(y),
		cluster(-1),
		minDist(FLT_MAX) {}

	float distance(Point p) {
		return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
	}
};

//store the data to csv file to run simulation in python
void generateOutputCSV(std::vector<Point>* points, bool gpuFlag) {
	std::string filePath = "../../../results/";
	std::string fileName;
	std::ofstream myfile;
	gpuFlag ? fileName = "outputGPU.csv" : fileName = "outputCPU.csv";
	myfile.open(filePath+fileName);
	myfile << "x,y,c" << std::endl;

	for (std::vector<Point>::iterator it = points->begin();
		it != points->end(); ++it) {
		myfile << it->x << "," << it->y << "," << it->cluster << std::endl;
	}
	myfile.close();
}

std::vector<Point> computePCA(std::vector<float> dataset1D, int numRows, int numCols)
{
	std::cout << "Reducing dimension to 2 using PCA...." << std::endl;
	std::vector<Point> pcaPoints;
	std::vector<float> pca1;
	// Form into a matrix.
	qbMatrix2<float> X(numRows, numCols, dataset1D);

	// Compute the covariance matrix.
	std::vector<float> columnMeans = qbPCA::ComputeColumnMeans(X);
	qbMatrix2<float> X2 = X;
	qbPCA::SubtractColumnMeans(X2, columnMeans);

	qbMatrix2<float> covX = qbPCA::ComputeCovariance(X2);

	// Compute the eigenvectors.
	qbMatrix2<float> eigenvectors;
	int testResult = qbPCA::ComputeEigenvectors(covX, eigenvectors);

	// Test the overall function.
	qbMatrix2<float> eigenvectors2;
	int testResult2 = qbPCA::qbPCA(X, eigenvectors2);

	// Test dimensionality reduction.
	qbMatrix2<float> V, part2;
	eigenvectors.Separate(V, part2, 2);

	qbMatrix2<float> newX = (V.Transpose() * X.Transpose()).Transpose();

	std::vector<std::vector<float>> pcaResult;
	std::vector<float > pcaRow;

	for (int i = 0; i < newX.GetNumRows(); ++i)
	{
		pcaRow.clear();
		for (int j = 0; j < newX.GetNumCols(); ++j)
		{
			pcaRow.push_back(newX.GetElement(i, j));
		}
		pcaResult.push_back(pcaRow);
	}

	for (int k = 0; k < pcaResult.size(); k++)
	{
		pcaPoints.push_back(Point(pcaResult[k][0], pcaResult[k][1]));
	}

	return pcaPoints;
}

std::vector<std::vector<float>> readCsv(std::string filename, bool hasLabels = true) {

	std::ifstream file(filename);
	// Make sure the file is open
	if(!file.is_open()) throw std::runtime_error("Could not open file");

	std::vector<std::vector<float>> pcadata;
	std::vector <std::string> uniqueWord;
	std::string line = "";
	std::string word = "";

	//skip the row containing the header line
	if(hasLabels)
		std::getline(file, line);
	
	while(std::getline(file, line)) {
		std::stringstream lineStream(line);
		std::vector<float> feature_data;

		while(lineStream.good()){
			std::string colField;			
			float value;

			std::getline(lineStream, colField, ',');
			try {
				value = std::stod(colField);
			}
			catch (std::invalid_argument e) {
				//if the data is not provided in the dataset, then replace it with float_min value
				if (colField == "") {
					value = FLT_MIN;
				}
				//if the dataset contains a column with string datatype, then this needs to be encoded
				else {
					bool isUnique = true;
					float encodedValue = 0;
					//run the loop for all unique words
					for (int k = 0; k < uniqueWord.size(); k++) {
						//check if the string is already present in the vector of unique words
						if (colField == uniqueWord[k]) {
							//if word already present in the unique words
							isUnique  = false;
							//encode the string with the value of the string(already present in uniqueWords) it matches with
							encodedValue = k;
							break;
						}
					}

					// check if the string is a unique word (new)
					if (isUnique) {
						//add it to the uniqueWord vector
						uniqueWord.push_back(colField);
						//encode the new string with a new number
						encodedValue = uniqueWord.size();
					}
					//save the categorical data into numerical form
					value = encodedValue;
				}
			}
			feature_data.push_back(value);
		}
		pcadata.push_back(feature_data);
	}

	std::vector<std::vector<float>> transContent;  //Vector to store the transpose of the content
	transContent = vect_Transpose(pcadata);

	//transpose the matrix and then fill the null values with average values
	int trowCount = transContent.size();
	int tcolCount = transContent[0].size();
	std::vector<float> validRowData; // data field with non NULL values
	std::vector<int> nullPosition;

	for (int i = 0; i < trowCount; i++)
	{
		nullPosition.clear();
		validRowData.clear();
		for (int j = 0; j < tcolCount; j++)
		{
			if (transContent[i][j] == FLT_MIN)
			{
				nullPosition.push_back(j);  //position of each FLT_MIN values in the column
			}

			else {
				validRowData.push_back(transContent[i][j]);
			}
		}

		if (!nullPosition.empty())
		{
			float sum = std::accumulate(validRowData.begin(), validRowData.end(), 0);
			float avg = sum / validRowData.size();
			for (auto nullPos = nullPosition.begin(); nullPos != nullPosition.end(); ++nullPos)
			{
				transContent[i][*nullPos] = avg;
			}
		}
	}

	//normalize data
	for (int j = 0; j < trowCount; j++) {
		std::vector<float> minmax;
		minmax = findMinMax(transContent[j]);
		for (int k = 0; k < transContent[j].size(); k++)
		{
			transContent[j][k] = (transContent[j][k] - minmax[0]) / (minmax[1] - minmax[0]);
		}
	}

	return vect_Transpose(transContent);;	
}

//kmeans main logic
int kMeansClustering(std::vector<Point>* points, std::vector<Point>* centroids, int epochs, int k) {	

	int counter=0;
	// initialise the clusters
	std::vector<Point> currCentroids;
	std::vector<Point> prevCentroids;

	for (int i = 0; i < k; ++i) {
		currCentroids.push_back((*centroids)[i]);
		prevCentroids = currCentroids;
	}

	//loop for finding better centroids
	for (int iterations = 0; iterations < epochs; iterations++) {
		std::vector<int> nPoints;
		std::vector<float> sumX, sumY;
		int similarCounter = 0;

		// Initialise with zeroes
		for (int j = 0; j < k; ++j) {
			nPoints.push_back(0);
			sumX.push_back(0.0);
			sumY.push_back(0.0);
		}

		//assign points to clusters
		for (std::vector<Point>::iterator it = points->begin();
				it != points->end(); ++it) {			
			Point p = *it;

			for (std::vector<Point>::iterator c = begin(currCentroids);
			c != end(currCentroids); ++c) {

				int clusterId = c - begin(currCentroids);
				float dist = c->distance(p);
				if (dist < p.minDist) {
					p.minDist = dist;
					p.cluster = clusterId;
				}
				*it = p;
			}
		}

		// Iterate over points to append data to centroids
		for (std::vector<Point>::iterator it = points->begin();
			it != points->end(); ++it) {
			int clusterId = it->cluster;
			nPoints[clusterId] += 1;
			sumX[clusterId] += it->x;
			sumY[clusterId] += it->y;

			it->minDist = FLT_MAX;  // reset distance
		}

		// Compute the new centroids
		for (std::vector<Point>::iterator c = begin(currCentroids);
			c != end(currCentroids); ++c) {
			
			int clusterId = c - begin(currCentroids);
			c->x = sumX[clusterId] / nPoints[clusterId];
			c->y = sumY[clusterId] / nPoints[clusterId];
			c->cluster = clusterId;
			if ((prevCentroids[clusterId].x == c->x) && (prevCentroids[clusterId].y == c->y)) {
				similarCounter ++;
			}
		}
		//stop the epochs when previous centroids == current centroids
		if (similarCounter == currCentroids.size()) {
			std::cout<<"stopped at epoch : "<<iterations<<std::endl;
			counter=iterations;
			break;
		}
		else
			prevCentroids = currCentroids;
	}

	std::cout<<"The final centroids generated by CPU : " <<std::endl;
	for(int i = 0; i < currCentroids.size(); i++) {
		std::cout<<"("<<currCentroids[i].x<<","<<currCentroids[i].y<<")"<<std::endl;
	}
	return counter;
}

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
	// Create a context
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[platformId] (), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);

	// Get the first device of the context
	std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size() << " devices" << std::endl;
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "../../../src/Kmeans.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Create a kernel object
	cl::Kernel kernel1(program, "assignCluster");
	cl::Kernel kernel2(program, "generateNewCentroids");

	// Get the maximum work group size for executing the kernel on the device
	size_t local;

	//clGetDeviceInfo();

	int epochs = 0;
	int noClusters = 0;
	std::string inputFile;

	std::cout << "Please enter the path of the dataset file:" << std::endl;
	std::cin >> inputFile;
	std::cout << "Please enter the maximum number of epochs you want to run:" << std::endl;
	std::cin >> epochs;
	std::cout << "Please enter the number of clusters:" << std::endl;
	std::cin >> noClusters;

	std::cout << "Reading csv file...." << std::endl;
	std::vector<std::vector<float>> dataSet = readCsv(inputFile);
	std::cout << "Csv file successfully imported...." << std::endl;
	std::cout << "Calculating total dimension of the given dataset...." << std::endl;


	int numRows = dataSet.size();
	int numCols = dataSet[0].size();
	std::vector<float> dataset1D;
	std::vector<Point> points, initialPoints;

	if (numCols > 2){
		for (int i = 0; i < numRows; i++)
		{
			for (int j = 0; j < numCols; j++)
			{
				dataset1D.push_back(dataSet[i][j]);
			}
		}

		initialPoints = computePCA(dataset1D, numRows, numCols);
	}
	else {
		for (int i = 0; i < numRows; i++)
		{
			initialPoints.push_back(Point(dataSet[i][0], dataSet[i][1]));
		}
	}

	points = initialPoints;

	std::cout << "Dataset converted to 2D..." << std::endl;
	srand(time(0));  // need to set the random seed
	std::vector<Point> initialCentroids;
	int n = initialPoints.size();
	for (int i = 0; i < noClusters; i++) {
		initialCentroids.push_back(initialPoints.at(rand() % n));
	}

	std::cout<<"Initial random centroids for GPU and CPU : "<<std::endl;
	for(int i = 0; i < noClusters; i++) {
		std::cout<<"("<<initialCentroids[i].x<<","<<initialCentroids[i].y<<")"<<std::endl;
	}
	
	// Declare some values
	std::size_t countPoints = n; // Overall number of work items = Number of elements
	std::size_t sizePoints = countPoints * sizeof (Point); // Size of data in bytes
	std::size_t sizeCentroids = noClusters * sizeof (Point);
	std::size_t sizeSum = noClusters * sizeof(float);
	std::size_t sizeNPoints = noClusters * sizeof(int);

	// Allocate space for input data and for output data from CPU and GPU on the host
	std::vector<float> sumX (noClusters);
	std::vector<float> sumY (noClusters);
	std::vector<int> nPoints (noClusters);
	std::vector<Point> newCentroids (noClusters); //final centroids


	// Allocate space for input and output data on the device
	cl::Buffer pointsBuffer = cl::Buffer (context, CL_MEM_READ_WRITE, sizePoints);
	cl::Buffer centroidsBuffer = cl::Buffer (context, CL_MEM_READ_WRITE, sizeCentroids);
	cl::Buffer sumXBuffer = cl::Buffer (context, CL_MEM_READ_WRITE, sizeSum);
	cl::Buffer sumYBuffer = cl::Buffer (context, CL_MEM_READ_WRITE, sizeSum);
	cl::Buffer nPointsBuffer = cl::Buffer (context, CL_MEM_READ_WRITE, sizeNPoints);

	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)

	std::cout << "Starting K means clustering on CPU...." << std::endl;
	Core::TimeSpan cpuStart = Core::getCurrentTime();
	int counter = kMeansClustering(&points, &initialCentroids, epochs, noClusters);
	Core::TimeSpan cpuEnd = Core::getCurrentTime();
	std::cout << "K means clustering completed on CPU...." << std::endl;

	std::cout << "Generating CSV file for CPU started" << std::endl;
	generateOutputCSV(&points, false);
	std::cout << "Generating CSV file for CPU finished" << std::endl;

	//reset points
	points = initialPoints;

	//Copy input data to device
	cl::Event copy1, copy2, copy3, copy4, copy5;
	queue.enqueueWriteBuffer(pointsBuffer, true, 0, sizePoints, points.data(), NULL, &copy1);
	queue.enqueueWriteBuffer(centroidsBuffer, true, 0, sizeCentroids, initialCentroids.data(), NULL, &copy2);
	queue.enqueueWriteBuffer(sumXBuffer, true, 0, sizeSum, sumX.data(), NULL, &copy3);
	queue.enqueueWriteBuffer(sumYBuffer, true, 0, sizeSum, sumY.data(), NULL, &copy4);
	queue.enqueueWriteBuffer(nPointsBuffer, true, 0, sizeNPoints, nPoints.data(), NULL, &copy5);

	cl::Event execution[2000];
	
	std::cout << "Starting K means clustering on GPU...." << std::endl;
	int count = 0;
	while(counter-- > 0){

		//Launch kernel1 on the device
		kernel1.setArg<cl::Buffer>(0, pointsBuffer);
		kernel1.setArg<cl::Buffer>(1, centroidsBuffer);
		kernel1.setArg<cl::Buffer>(2, sumXBuffer);
		kernel1.setArg<cl::Buffer>(3, sumYBuffer);
		kernel1.setArg<cl::Buffer>(4, nPointsBuffer);
		kernel1.setArg<int>(5, noClusters);

		queue.enqueueNDRangeKernel(kernel1, 0, countPoints, 1, NULL, &execution[count++]);
		
		//Launch kernel2 on the device
		kernel2.setArg<cl::Buffer>(0, centroidsBuffer);
		kernel2.setArg<cl::Buffer>(1, sumXBuffer);
		kernel2.setArg<cl::Buffer>(2, sumYBuffer);
		kernel2.setArg<cl::Buffer>(3, nPointsBuffer);

		queue.enqueueNDRangeKernel(kernel2, 0, noClusters, 1, NULL, &execution[count++]);

	}

	// Copy output data back to host
	cl::Event copy6;
	queue.enqueueReadBuffer(centroidsBuffer, true, 0, sizeCentroids, newCentroids.data(), NULL, &copy6);
	
	cl::Event copy7;
	queue.enqueueReadBuffer(pointsBuffer, true, 0, sizePoints, points.data(), NULL, &copy7);

	std::cout << "Generating CSV file for CPU started" << std::endl;
	generateOutputCSV(&points, true);
	std::cout << "Generating CSV file for CPU finished" << std::endl;

	std::cout<<"The final centroids generated by GPU : "<<std::endl;
	for(int i=0; i< noClusters; i++) {
		std::cout<<"("<<newCentroids[i].x<<","<<newCentroids[i].y<<")"<<std::endl;
	}

	std::cout << "K means clustering completed on GPU...." << std::endl;
	Core::TimeSpan cpuTime = cpuEnd - cpuStart;

	Core::TimeSpan copyTime1 = OpenCL::getElapsedTime(copy1);
	Core::TimeSpan copyTime2 = OpenCL::getElapsedTime(copy2);
	Core::TimeSpan copyTime3 = OpenCL::getElapsedTime(copy3);
	Core::TimeSpan copyTime4 = OpenCL::getElapsedTime(copy4);
	Core::TimeSpan copyTime5 = OpenCL::getElapsedTime(copy5);
	Core::TimeSpan copyTime6 = OpenCL::getElapsedTime(copy6);
	Core::TimeSpan copyTime7 = OpenCL::getElapsedTime(copy7);
	Core::TimeSpan copyTime = copyTime1 + copyTime2 + copyTime3 + copyTime4 + copyTime5 + copyTime6 + copyTime7;

	Core::TimeSpan gpuTime = Core::TimeSpan::fromSeconds(0);
	for (std::size_t i = 0; i < count; i++)
		gpuTime = gpuTime + OpenCL::getElapsedTime(execution[i]);

	Core::TimeSpan overallGpuTime = gpuTime + copyTime;

	std::cout << "CPU Time: " << cpuTime.toString() << std::endl;
	std::cout << "Memory copy Time: " << copyTime.toString() << std::endl;
	std::cout << "GPU Time w/o memory copy: " << gpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds()) << ")" << std::endl;
	std::cout << "GPU Time with memory copy: " << overallGpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / overallGpuTime.getSeconds()) << ")" << std::endl;

	std::cout << "Success" << std::endl;

	return 0;
}
