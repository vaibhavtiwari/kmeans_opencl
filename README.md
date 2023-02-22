# kmeans_opencl

Flowchart of K-means clustering:

<img src=https://user-images.githubusercontent.com/84077845/220650466-4fadc993-4c56-4017-9b2e-8a8de1d4ce7b.png width="400" height="600">

Input Example: 

<img src=https://user-images.githubusercontent.com/84077845/220650631-d1143c32-9375-44b5-b42a-ea9ca4b088e6.png width="400" height="200">

Output Example:
1.	Execution
a.	Debug

<img src=https://user-images.githubusercontent.com/84077845/220650749-1b341a4f-62b2-4e69-89c0-a029c448f5af.png width="400" height="200">

b.	Release

<img src=https://user-images.githubusercontent.com/84077845/220650829-d29aeebc-5161-4629-8295-735bd28af8b6.png width="400" height="200">

2.	Visualization Example:

<img src=https://user-images.githubusercontent.com/84077845/220650930-bc445cde-eb9a-467f-a65e-594955eed37f.png width="400" height="200">


Steps To Execute:
1.	Code:
a.	Run The Kmeans.cpp file.
b.	Provide the path of the datasets, number of epochs, and number of clusters.
c.	The output files will be generated and stored in the folder \results and speed-up can be seen on the cmd.
2.	Visualization:
a.	Run the output_generator.ipynb in jupyter notebook at the given location (\results).


References:
PCA	QuantitativeBytes, “qbLinAlg”. [Online].  Available:  QuantitativeBytes/qbLinAlg: QuantitativeBytes Linear Algebra Library [C++]. A simple implementation of various common linear algebra functions, intended for educational purposes. (github.com), 2021

Sample Data Generation	K.Arvai, “K-Means Clustering in Python: A Practical Guide” [Online].  Available: https://realpython.com/k-means-clustering-python/#:~:text=The%20k%2Dmeans%20clustering%20method,the%20oldest%20and%20most%20approachable

K-means CPU code	Robert Andrew Martin, “Implementing k-means clustering from scratch in C++”. [Online].  Available: https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/, 2019


