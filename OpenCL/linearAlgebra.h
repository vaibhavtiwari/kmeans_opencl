
//////////////////////////////////////////////////////////////////////////////
// Principal Component Analysis Implementation
//////////////////////////////////////////////////////////////////////////////

// Define error codes.
constexpr int PCA_MATRIXNOTSQUARE = -1;
constexpr int PCA_MATRIXNOTSYMMETRIC = -2;

//Check whether a matrix is a symmetric matrix or not.
template <typename T>
bool IsSymmetric(const vector<vector<T>>& inputData)
{
	vector<vector<T>> transData;
	transData = vect_Transpose(inputData);
	int numRows = inputData.size();
	int numCols = inputData[0].size();

	for (int i = 0; i < numRows.size(); i++)
	{
		for (int j = 0; j < numCols; j += )
		{
			if (inputData[i][j] != transData[i][j])
			{
				return false;
			}
		}
	}

	return true;
}

//Check whether a matrix is a square matrix or not.
template <typename T>
bool IsSquare(const vector<vector<T>>& inputData)
{
	if (inputData.size() == inputData[0].size())
	{
		return true;
	}
	else
	{
		return false;
	}
}

template <class T>
bool CloseEnough(T f1, T f2)
{
	return fabs(f1 - f2) < 1e-9;
}

template <class T>
bool IsRowEchelon(vector<vector<T>> inputData)
{
	m_nRows = inputData.size();
	/* We do this by testing that the sum of all the elements in the
		lower triangular matrix is zero. */
		// Loop over each row, except the first one (which doesn't need to have any zero elements).
	T cumulativeSum = static_cast<T>(0.0);
	for (int i = 1; i < m_nRows; ++i)
	{
		/* Loop over the columns that correspond to the lower triangular
			matrix according to the current row. */
		for (int j = 0; j < i; ++j)
		{
			// Add this element to the cumulative sum.
			cumulativeSum += m_matrixData[Sub2Ind(i, j)];
		}
	}

	/* If the matrix is in row-echelon form, then cumulative sum should
		still equal zero, otherwise the matrix cannot be in row-echelon form. */
	return CloseEnough(cumulativeSum, 0.0);
}

//This function creates Identity Matrix.
template <typename T>
vector<vector<T>> createIdentityMatrix(int numRows)
{
	vector<vector<T>> output;
	vector<T> rowData;

	for (int i = 0; i < numRows; i++)
	{
		rowData.clear();
		for (int j = 0; j < numRows; j++)
		{
			if (i == j)
			{
				rowData.push_back(1);
			}
			else
			{
				rowData.push_back(0);
			}
		}
		output.push_back(rowData);
	}
	return output;
}



//This function returns the mean of each column 
//input: data with of vector mxn size where m:number of observations  n:number of variables
//output: vector of p size where p: number of variables
template <typename T>
vector<T> ComputeColumnMeans(const vector<vector<T>>& inputData)
{
	// Determine the size of the input data.
	int numRows = inputData.GetNumRows();
	int numCols = inputData[0].GetNumCols();
	double sum;
	double mean;
	// Create a vector for output.
	vector<T> output;

	// Loop through and compute means.
	vector<vector<T>> transData = vect_Transpose(inputData);
	int numObserv = transData[i].size();

	for (int i = 0; i < transData.size(); i++)
	{
		sum = accumulate(transData[i].begin(), transData[i].end(), 0);
		mean = sum / (numObserv);
		output.push_back(mean);
	}
	return output;
}


// Function to subtract the column means.
template <typename T>
void SubtractColumnMeans(vector<vector<T>>& inputData, vector<T>& columnMeans)
{
	// Determine the size of the input data.
	int numRows = inputData.size();
	int numCols = inputData[0].size();

	// Loop through and subtract the means.
	if (numCols == columnMeans[size])
	{
		for (int j = 0; j < numCols; ++j)
		{
			for (int i = 0; i < numRows; ++i)
				inputData[i][j] = inputData[i][j] - columnMeans.at(j);
		}
	}
	else
	{
		cout << "Mismatch: Number of variables in Input Data and Mean data." << endl;
	}

}

// Function to compute the covaraince matrix.
template <typename T>
vector<vector<T>> ComputeCovariance(const vector<vector<T>>& inputData)
{
	/* Compute the covariance matrix.
		Note that here we use X'X, rather than XX' as is the usual case.
		This is because we are requiring our data to be arranged with one
		column (p) for each variable, with one row (k) for each observation. If
		we computed XX', the result would be a [k x k] matrix. The covariance
		matrix should be [p x p], so we need to transpose, hence the use of
		X'X. */

	int numRows = inputData.size();
	vector<vector<T>> covMatrix = (static_cast<T>(1.0) / static_cast<T>(numRows - 1)) * (vect_Transpose(inputData) * inputData);
	return covMatrix;
}

// Compute the length of the vector,known as the 'norm'.
template <class T>
T norm(vector<T> inputData)
{
	T cumulativeSum = static_cast<T>(0.0);
	for (int i = 0; i < inputData.size(); ++i)
		cumulativeSum += (inputData.at(i) * inputData.at(i));

	return sqrt(cumulativeSum);
}

// Return a normalized copy of the vector.
template <class T>
vector<T> Normalized(vector<T> data)
{
	// Compute the vector norm.
	T vecNorm = norm(data);

	// Compute the normalized version of the vector.
	vector<T> result(data.begin(), data.end());
	return result * (static_cast<T>(1.0) / vecNorm);
}

// Normalize the vector in place.
template <class T>
void Normalize(vector<T>& input)
{
	// Compute the vector norm.
	T vecNorm = norm(input);

	// Compute the elements of the normalized version of the vector.
	for (int i = 0; i < m_nDims; ++i)
	{
		T temp = input.at(i) * (static_cast<T>(1.0) / vecNorm);
		input.at(i) = temp;
	}
}

/* **************************************************************************************************
JOIN TwO MATRICES TOGETHER
/* *************************************************************************************************/
template <class T>
bool Join(vector<vector<T>>& matrix1, const vector<vector<T>>& matrix2)
{
	// Extract the information that we need from both matrices
	int numRows1 = matrix1.size();
	int numRows2 = matrix2.size();
	int numCols1 = matrix1[0].size();
	int numCols2 = matrix2[0].size();

	// If the matrices have different numbers of rows, then this operation makes no sense.
	if (numRows1 != numRows2)
		throw invalid_argument("Attempt to join matrices with different numbers of rows is invalid.");

	// Allocate memory for the result.
	// Note that only the number of columns increases.
	T* newMatrixData = new T[numRows1 * (numCols1 + numCols2)];

	// Copy the two matrices into the new one.
	int linearIndex, resultLinearIndex;
	for (int i = 0; i < numRows1; ++i)
	{
		for (int j = 0; j < (numCols1 + numCols2); ++j)
		{
			resultLinearIndex = (i * (numCols1 + numCols2)) + j;

			// If j is in the left hand matrix, we get data from there...
			if (j < numCols1)
			{
				linearIndex = (i * numCols1) + j;
				newMatrixData[resultLinearIndex] = matrix1[linearIndex];
			}
			// Otherwise, j must be in the right hand matrix, so we get data from there...
			else
			{
				linearIndex = (i * numCols2) + (j - numCols1);
				newMatrixData[resultLinearIndex] = matrix2.m_matrixData[linearIndex];
			}
		}
	}

	// Update the stored data.
	m_nCols = numCols1 + numCols2;
	m_nElements = m_nRows * m_nCols;
	delete[] m_matrixData;
	m_matrixData = new T[m_nElements];
	for (int i = 0; i < m_nElements; ++i)
		m_matrixData[i] = newMatrixData[i];

	delete[] newMatrixData;
	return true;
}

/* **************************************************************************************************
COMPUTE MATRIX INVERSE (USING GAUSS-JORDAN ELIMINATION)
/* *************************************************************************************************/
template <class T>
bool Inverse(vector<vector<T>>& input)
{
	// Check if the matrix is square (we cannot compute the inverse if it isn't).
	if (!IsSquare(input))
		throw invalid_argument("Cannot compute the inverse of a matrix that is not square.");

	// If we get to here, the matrix is square so we can continue.

	// Form an identity matrix with the same dimensions as the matrix we wish to invert.
	vector<vector<T>> identityMatrix = createIdentityMatrix(input.size());

	// Join the identity matrix to the existing matrix.	
	int originalNumCols = m_nCols;
	Join(identityMatrix);

	// Begin the main part of the process.
	int cRow, cCol;
	int maxCount = 100;
	int count = 0;
	bool completeFlag = false;
	while ((!completeFlag) && (count < maxCount))
	{
		for (int diagIndex = 0; diagIndex < m_nRows; ++diagIndex)
		{
			// Loop over the diagonal of the matrix and ensure all diagonal elements are equal to one.
			cRow = diagIndex;
			cCol = diagIndex;

			// Find the index of the maximum element in the current column.
			int maxIndex = FindRowWithMaxElement(cCol, cRow);

			// If this isn't the current row, then swap.
			if (maxIndex != cRow)
			{
				//std::cout << "Swap rows " << cRow << " and " << maxIndex << std::endl;
				SwapRow(cRow, maxIndex);
			}
			// Make sure the value at (cRow,cCol) is equal to one.
			if (m_matrixData[Sub2Ind(cRow, cCol)] != 1.0)
			{
				T multFactor = 1.0 / m_matrixData[Sub2Ind(cRow, cCol)];
				MultRow(cRow, multFactor);
				//std::cout << "Multiply row " << cRow << " by " << multFactor << std::endl;
			}

			// Consider the column.
			for (int rowIndex = cRow + 1; rowIndex < m_nRows; ++rowIndex)
			{
				// If the element is already zero, move on.
				if (!CloseEnough(m_matrixData[Sub2Ind(rowIndex, cCol)], 0.0))
				{
					// Obtain the element to work with from the matrix diagonal.
					// As we aim to set all the diagonal elements to one, this should
					// always be valid for a matrix that can be inverted.
					int rowOneIndex = cCol;

					// Get the value stored at the current element.
					T currentElementValue = m_matrixData[Sub2Ind(rowIndex, cCol)];

					// Get the value stored at (rowOneIndex, cCol)
					T rowOneValue = m_matrixData[Sub2Ind(rowOneIndex, cCol)];

					// If this is equal to zero, then just move on.
					if (!CloseEnough(rowOneValue, 0.0))
					{
						// Compute the correction factor.
						// (required to reduce the element at (rowIndex, cCol) to zero).
						T correctionFactor = -(currentElementValue / rowOneValue);

						MultAdd(rowIndex, rowOneIndex, correctionFactor);

						//std::cout << "Multiply row " << rowOneIndex << " by " << correctionFactor <<
						//	" and add to row " << rowIndex << std::endl;
					}
				}
			}

			// Consider the row.			
			for (int colIndex = cCol + 1; colIndex < originalNumCols; ++colIndex)
			{
				// If the element is already zero, move on.
				if (!CloseEnough(m_matrixData[Sub2Ind(cRow, colIndex)], 0.0))
				{
					// Obtain the element to work with from the matrix diagonal.
					// As we aim to set all the diagonal elements to one, this should
					// always be valid for a matrix that can be inverted.
					int rowOneIndex = colIndex;

					// Get the value stored at the current element.
					T currentElementValue = m_matrixData[Sub2Ind(cRow, colIndex)];

					// Get the value stored at (rowOneIndex, colIndex)
					T rowOneValue = m_matrixData[Sub2Ind(rowOneIndex, colIndex)];

					// If this is equal to zero, then just move on.
					if (!CloseEnough(rowOneValue, 0.0))
					{

						// Compute the correction factor.
						// (required to reduce the element at (cRow, colIndex) to zero).
						T correctionFactor = -(currentElementValue / rowOneValue);

						// To make this equal to zero, we need to add -currentElementValue multiplied by
						// the row at rowOneIndex.
						MultAdd(cRow, rowOneIndex, correctionFactor);

						//std::cout << "Multiply row " << rowOneIndex << " by " << correctionFactor <<
						//	" and add to row " << cRow << std::endl;
					}
				}
			}
		}

		// Separate the result into the left and right halves.
		qbMatrix2<T> leftHalf;
		qbMatrix2<T> rightHalf;
		this->Separate(leftHalf, rightHalf, originalNumCols);

		// When the process is complete, the left half should be the identity matrix.
		if (leftHalf == identityMatrix)
		{
			// Set completedFlag to true to indicate that the process has completed.
			completeFlag = true;

			// Rebuild the matrix with just the right half, which now contains the result.			
			m_nCols = originalNumCols;
			m_nElements = m_nRows * m_nCols;
			delete[] m_matrixData;
			m_matrixData = new T[m_nElements];
			for (int i = 0; i < m_nElements; ++i)
				m_matrixData[i] = rightHalf.m_matrixData[i];
		}

		// Increment the counter.
		count++;
	}

	// Return whether the process succeeded or not.
	return completeFlag;
}


// The qbQR function.
template <typename T>
int QRDecomposition(const vector<vector<T>>& A, vector<vector<T>>& Q, vector<vector<T>>& R)
{

	// Make a copy of the input matrix.
	vector<vector<T>> inputData = A;

	// Verify that the input matrix is square.
	if (!inputData.IsSquare())
		return QBQR_MATRIXNOTSQUARE;

	// Determine the number of columns (and rows, since the matrix is square).
	int numCols = inputData[0].size();

	// Create a vector to store the P matrices for each column.
	vector<vector<vector<T>>> Plist;

	// Loop through each column.
	for (int j = 0; j < (numCols - 1); ++j)
	{
		// Create the a1 and b1 vectors.
		// a1 is the column vector from A.
		// b1 is the vector onto which we wish to reflect a1.
		vector<T> a1(numCols - j);
		vector<T> b1(numCols - j);

		for (int i = j; i < numCols; ++i)
		{
			a1[i - j] = inputData[i][j];
			b1[i - j] = static_cast<T>(0.0);

			/*a1.SetElement(i - j, inputMatrix.GetElement(i, j));
			b1.SetElement(i - j, static_cast<T>(0.0));*/
		}
		b1[0] = static_cast<T>(1.0);
		/*b1.SetElement(0, static_cast<T>(1.0));*/

		// Compute the norm of the a1 vector.
		T a1norm = norm(a1);

		// Compute the sign we will use.
		int sgn = -1;
		if (a1[0] < static_cast<T>(0.0))
			sgn = 1;

		// Compute the u-vector.
		vector<T> u = a1 - (sgn * a1norm * b1);

		// Compute the n-vector.
		vector<T> n = Normalized(u);

		// Convert n to a matrix so that we can transpose it.
		vector<vector<T>> nMat(numCols - j, vector <T>(1);
		for (int i = 0; i < (numCols - j); ++i)
			nMat[i][0] = n[i];
		/*nMat.SetElement(i, 0, n.GetElement(i));*/

	// Transpose nMat.
		vector<vector<T>> nMatT = vect_Transpose(nMat);

		// Create an identity matrix of the appropriate size.
		vector<vector<T>> I = createIdentityMatrix(numCols - j);

		// Compute Ptemp.
		vector<vector<T>> Ptemp = I - static_cast<T>(2.0) * nMat * nMatT;

		// Form the P matrix with the original dimensions.
		vector<vector<T>> P = createIdentityMatrix(numCols)

			for (int row = j; row < numCols; ++row)
			{
				for (int col = j; col < numCols; ++col)
				{
					P[row][col] = Ptemp[row - j][col - j];
				}
			}

		// Store the result into the Plist vector.
		Plist.push_back(P);

		// Apply this transform matrix to inputMatrix and use this result
		// next time through the loop.
		inputData = P * inpuData;
	}

	// Compute Q.
	vector<vector<T>> Qmat = Plist.at(0);
	for (int i = 1; i < (numCols - 1); ++i)
	{
		Qmat = Qmat * vect_Transpose(Plist.at(i));
	}

	// Return the Q matrix.
	Q = Qmat;

	// Compute R.
	int numElements = Plist.size();
	vector<vector<T>> Rmat = Plist.at(numElements - 1);
	for (int i = (numElements - 2); i >= 0; --i)
	{
		Rmat = Rmat * Plist.at(i);
	}
	Rmat = Rmat * A;

	// And return the R matrix.
	R = Rmat;

}


// Function to estimate (real) eigenvalues using QR decomposition.
/* Note that this is only valid for matrices that have ALL real
	eigenvalues. The only matrices that are guaranteed to have only
	real eigenvalues are symmetric matrices. Therefore, this function
	is only guaranteed to work with symmetric matrices. */
template <typename T>
int computeEigenValues(const vector<vector<T>>& inputData, vector<T>& eigenValues)
{
	// Make a copy of the input matrix.
	vector<vector<T>> A = inputMatrix;

	// Verify that the input matrix is square.
	if (!IsSquare(A))
		return QBEIG_MATRIXNOTSQUARE;

	// Verify that the matrix is symmetric.
	if (!IsSymmetric(A))
		return QBEIG_MATRIXNOTSYMMETRIC;

	// The number of eigenvalues is equal to the number of rows.
	int numRows = data.size();

	// Create an identity matrix of the same dimensions.
	vector<vector<T>> identityMatrix = createIdentityMatrix(numRows);

	// Create matrices to store Q and R.
	vector<vector<T>> Q(numRows, vector<T>(numRows));
	vector<vector<T>> R(numRows, vector<T>(numRows));

	// Loop through each iteration.
	int maxIterations = 10e3;
	int iterationCount = 0;
	bool continueFlag = true;
	while ((iterationCount < maxIterations) && continueFlag)
	{
		// Compute the QR decomposition of A.
		int returnValue = QRDecomposition(data, Q, R);

		// Compute the next value of A as the product of R and Q.
		A = R * Q;

		/* Check if A is now close enough to being upper-triangular.
			We can do this using the IsRowEchelon() function from the
			qbMatrix2 class. */
		if (IsRowEchelon(A))
			continueFlag = false;

		// Increment iterationCount.
		iterationCount++;
	}

	// At this point, the eigenvalues should be the diagonal elements of A.
	for (int i = 0; i < numRows; ++i)
		eigenValues.push_back(A[i][i]);

	// Set the return status accordingly.
	if (iterationCount == maxIterations)
		return QBEIG_MAXITERATIONSEXCEEDED;
	else
		return 0;

}

// Function to perform inverse power iteration method.
template <typename T>
int InvPIt(const vector<vector<T>>& inputMatrix, const T& eigenValue, vector<T>& eigenVector)
{
	// Make a copy of the input matrix.
	vector<vector<T>> A = inputMatrix;

	// Verify that the input matrix is square.
	if (!IsSquare(A))
		return QBEIG_MATRIXNOTSQUARE;

	// Setup a random number generator.
	random_device myRandomDevice;
	mt19937 myRandomGenerator(myRandomDevice());
	uniform_int_distribution<int> myDistribution(1.0, 10.0);

	/* The number of eigenvectors and eigenvalues that we will compute will be
		equal to the number of rows in the input matrix. */
	int numRows = A.size();

	// Create an identity matrix of the same dimensions.
	vector<vector<T>> identityMatrix = createIdentityMatrix(numRows);

	// Create an initial vector, v.
	vector<T> v(numRows);
	for (int i = 0; i < numRows; ++i)
		v[i] = static_cast<T>(myDistribution(myRandomGenerator));

	// Iterate.
	int maxIterations = 100;
	int iterationCount = 0;
	T deltaThreshold = static_cast<T>(1e-9);
	T delta = static_cast<T>(1e6);
	vector<T> prevVector(numRows);
	vector<vector<T>> tempMatrix(numRows, vector<T>(numRows));

	while ((iterationCount < maxIterations) && (delta > deltaThreshold))
	{
		// Store a copy of the current working vector to use for computing delta.
		prevVector = v;

		// Compute the next value of v.
		tempMatrix = A - (eigenValue * identityMatrix);
		tempMatrix.Inverse();
		v = tempMatrix * v;
		Normalize(v);

		// Compute delta.
		delta = norm((v - prevVector));

		// Increment iteration count.
		iterationCount++;
	}

	// Return the estimated eigenvector.
	eigenVector = v;

	// Set the return status accordingly.
	if (iterationCount == maxIterations)
		return QBEIG_MAXITERATIONSEXCEEDED;
	else
		return 0;

}


// Function to compute the eigenvectors of the covariance matrix.
template <typename T>
int ComputeEigenvectors(const vector<vector<T>>& covarianceMatrix, vector<vector<T>>& eigenvectors)
{
	// Copy the input matrix.
	vector<vector<T>> X = covarianceMatrix;

	// The covariance matrix must be square and symmetric.
	if (!IsSquare(X))
		return PCA_MATRIXNOTSQUARE;

	// Verify that the matrix is symmetric.
	if (!IsSymmetric(X))
		return PCA_MATRIXNOTSYMMETRIC;

	// Compute the eignvalues.
	vector<T> eigenValues;
	int returnStatus = computeEigenValues(X, eigenValues);

	// Sort the eigenvalues.
	sort(eigenValues.begin(), eigenValues.end());
	reverse(eigenValues.begin(), eigenValues.end());

	// Compute the eigenvector for each eigenvalue.
	vector<T> eV(X[0].size());
	vector<vector<T>> eVM(X.size(), vector<T>(X[0]size()));
	for (int j = 0; j < eigenValues.size(); ++j)
	{
		T eig = eigenValues.at(j);
		int returnStatus2 = qbInvPIt<T>(X, eig, eV);
		for (int i = 0; i < eV.GetNumDims(); ++i)
			eVM.SetElement(i, j, eV.GetElement(i));
	}

	// Return the eigenvectors.
	eigenvectors = eVM;

	// Return the final return status.	
	return returnStatus;
}
