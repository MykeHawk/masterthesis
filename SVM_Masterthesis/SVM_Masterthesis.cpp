// SVM_Masterthesis.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <iterator>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type)) //easier way to perform a malloc
using namespace std;

//Training or testing with labels
int processFile(ifstream& myfile, vector<double>& yLabels, vector<int>&indexNodes, vector<int>& valueNodes, int& problemLength);
//Testing without labels
//int processTestFile(ifstream& myfile2, vector<int>& indexTestNodes, vector<int>& valueTestNodes, int& testRows);

int main()
{
	/* Reading training file*/
	ifstream myfile("a1a_train.txt");
	vector<double> yLabels;
	vector<int> indexNodes;
	vector<int> valueNodes;
	int problemLength = 0;
	int success = processFile(myfile, yLabels, indexNodes, valueNodes, problemLength);
	if (success != 0)
	{
		//Processing file failed
		cout << "Something went wrong with processing the training file!" << endl;
		cerr << "Error: " << strerror(errno);
		cin.get();
		return 1;
	}
	//read in test data -----------------------------------------------------------------------
	ifstream myfile2("a1a.txt");
	vector<double> yTestLabels;
	vector<int> indexTestNodes;
	vector<int> valueTestNodes;
	int testRows = 0;
	success = processFile(myfile2, yTestLabels, indexTestNodes, valueTestNodes, testRows);
	if (success != 0)
	{
		//Processing file failed
		cout << "Something went wrong with processing the test file!" << endl;
		cerr << "Error: " << strerror(errno);
		cin.get();
		return 1;
	}
	//-------------------------------------------------------------------------------------------------------------------------------
	/* Setting up Parameter for model
	Initialize every parameter for future reasons */
	svm_parameter param;
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0.5;
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.eps = 0.001;
	param.C = 1;
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.probability = 0;
	param.shrinking = 1;
	//-------------------------------------------------------------------------------------------------------------------------------
	//Setting up the problem for model --> Training our data
	svm_problem problem;
	problem.l = problemLength;
	cout << "Number of rows for the problem: " << problem.l << endl;
	int rowProblem = 0;
	int numberOfColumns = ceil(indexNodes.size() / problem.l);

	cout << "number of column = " << numberOfColumns << " number of rows = " << problem.l << endl;
	/*
	Generate the data for our problem on each row of our training data
	*/
	svm_node **x = Malloc(svm_node*, problem.l);
	int indexCounter = 0;
	int valueCounter = 0;
	int col = 0;
	for (int row = 0; row < problem.l; row++)
	{
		svm_node * x_space = Malloc(svm_node, numberOfColumns);
		while (indexNodes[indexCounter] != -1)
		{
			x_space[col].index = indexNodes[indexCounter];
			x_space[col].value = valueNodes[valueCounter];
			valueCounter++;
			indexCounter++;
			col++;
		}
		x_space[col].index = -1;
		indexCounter++;
		col = 0;
		x[row] = x_space;
	}
	cout << indexCounter << " " << valueCounter << " " << indexNodes.size() << " " << valueNodes.size() << endl;
	/*
	Generate the correct labels given to each row
	*/
	problem.x = x;
	problem.y = Malloc(double, problem.l);
	int labelCount = 0;
	for (vector<double>::iterator it = yLabels.begin(); it != yLabels.end(); ++it)
	{
		problem.y[labelCount] = *it;
		labelCount++;
	}
	//Create a model by training a problem and parameter
	svm_model *model = svm_train(&problem, &param);
	cout << "Training done! Number of rows in model = " << problemLength << endl;
	if (svm_check_parameter(&problem, &param) != NULL)
	{
		cout << svm_check_parameter(&problem, &param) << endl;
	}
	//-------------------------------------------------------------------------------------------------------------------------------
	//Create test node with values from a1a, later on we iterate over this test node to perform our SVM classification
	svm_node **testNode = Malloc(svm_node*, testRows);
	indexCounter = 0;
	valueCounter = 0;
	col = 0;
	int testColumns = ceil(indexTestNodes.size() / testRows);
	for (int row = 0; row < testRows; row++)
	{
		svm_node * testNode_space = Malloc(svm_node, numberOfColumns);
		while (indexTestNodes[indexCounter] != -1)
		{
			testNode_space[col].index = indexTestNodes[indexCounter];
			testNode_space[col].value = valueTestNodes[valueCounter];
			valueCounter++;
			indexCounter++;
			col++;
		}
		testNode_space[col].index = -1;
		indexCounter++;
		col = 0;
		testNode[row] = testNode_space;
	}
	//-------------------------------------------------------------------------------------------------------------------------------
	//retval returns the correct label (Y-value) of the corresponding row that matches the data
	/*double retval = svm_predict(model, testNode[25]); 
	cout << "Return value = " << retval << endl;
	cout << "Value from line 26 on test node: " << testNode[24][0].index << " With label: " << yTestLabels[24] << endl;*/
	/* 
	Returning the accuracy of the giving training file with a certain test file
	Accuracy shows how many Test nodes were accurately predicted towards their given class (supervised learning)
	*/
	double accuracy = 0;
	int totalClassifications = 0;
	double retval = 0;
	vector<int> returnValues;
	for (int row = 0; row < testRows; row++)
	{
		retval = svm_predict(model, testNode[row]);
		returnValues.push_back(retval);
		if (retval == yTestLabels[row]) ++accuracy;
		++totalClassifications;
	}
	cout << "Return value of prediction = " << retval << endl;
	cout << "Accuracy = " << accuracy / totalClassifications * 100 << "% (" << accuracy << "/" << totalClassifications << ")" << endl;

	/*
	Display what the return value was for a given row on the file
	*/
	cout << "Displaying return values for the given training file:" << endl;
	int returnRow = 1;
	for (vector<int>::iterator it = returnValues.begin(); it != returnValues.end(); ++it)
	{
		cout << "Row " << returnRow << ": " << *it << endl;
		++returnRow;
	}
	
	std::ofstream output_file("./returnValues.txt");
	std::ostream_iterator<int> output_iterator(output_file, "\n");
	std::copy(returnValues.begin(), returnValues.end(), output_iterator);
	
	/* DESTROY ALL DATA */
	svm_destroy_param(&param);


	cin.get();
    return 0;
}

int processFile(ifstream& myfile, vector<double>& yLabels, vector<int>&indexNodes, vector<int>& valueNodes, int& problemLength)
{
	string line;
	string lineSubstring;
	int position = 0;
	int positionSubstring = 0;
	int length = 0;
	int index = 0;
	int value = 0;
	char spaceChar;
	double y = 0;

	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			problemLength++;
			position = 0;
			for (int i = 0; i < line.length(); i++)
			{
				spaceChar = line[i];
				if (isspace(spaceChar))
				{

					//read every part between " " and the next " "
					lineSubstring = line.substr(position, i - position);
					if (position == 0) //get all labels
					{
						//conversion of 1 or -1 to double
						y = atof(lineSubstring.c_str());
						yLabels.push_back(y);
					}
					else //get nodes of a certain label
					{
						positionSubstring = lineSubstring.find(":");
						index = stoi(lineSubstring.substr(0, positionSubstring));
						value = stoi(lineSubstring.substr(positionSubstring + 1, lineSubstring.length()));
						indexNodes.push_back(index);
						valueNodes.push_back(value);
					}

					position = i + 1;
				}
			}
			//Indicate end of x node
			indexNodes.push_back(-1);
		}
		myfile.close();
		return 0;
	}
	else
	{
		cout << "Unable to open file" << endl;
		return 1;
	}

}

/*int processTestFile(ifstream& myfile2, vector<int>& indexTestNodes, vector<int>& valueTestNodes, int& testRows)
{
	string line;
	string lineSubstring;
	int position = 0;
	int positionSubstring = 0;
	int length = 0;
	int index = 0;
	int value = 0;
	char spaceChar;
	double y = 0;

	if (myfile2.is_open())
	{
		while (getline(myfile2, line))
		{
			testRows++;
			position = 0;
			for (int i = 0; i < line.length(); i++)
			{
				spaceChar = line[i];
				if (isspace(spaceChar))
				{

					//read every part between " " and the next " "
					lineSubstring = line.substr(position, i - position);
					if (position == 0) //get all labels
					{
						//conversion of 1 or -1 to double
						y = atof(lineSubstring.c_str());
					}
					else //get nodes of a certain label
					{
						positionSubstring = lineSubstring.find(":");
						index = stoi(lineSubstring.substr(0, positionSubstring));
						value = stoi(lineSubstring.substr(positionSubstring + 1, lineSubstring.length()));
						indexTestNodes.push_back(index);
						valueTestNodes.push_back(value);
					}

					position = i + 1;
				}
			}
			//indicate end of node
			indexTestNodes.push_back(-1);
		}
		myfile2.close();
		return 0;
	}
	else
	{
		cout << "Unable to open file" << endl;
		return 1;
	}
} */

