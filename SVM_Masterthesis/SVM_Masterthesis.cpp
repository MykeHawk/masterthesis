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
#include "datafile.h"
#include "trainer.h"
#include "Model.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type)) //easier way to perform a malloc
using namespace std;

int main()
{
	/* Testing class datafile*/
	DataFile training_file("a1a_train.txt");
	svm_problem svm_test_problem = training_file.getProblem();
//	cout << "testing svm problem from class: " << svm_test_problem.x[27][0].index << endl;

	DataFile test_node_datafile("a1a.txt");
	svm_node ** test_node = test_node_datafile.getNode();
//	cout << "testing test node: " << test_node[27][0].index << endl;

	Model test_model;
	Trainer test_trainer;

	test_trainer.train(test_model, training_file);

	DataFile test_model_datafile = test_model.getDatafile();
	svm_node ** test_datafile_node = test_model_datafile.getNode();
	cout << "++testing datafile node from trained model: " << test_datafile_node[27][0].index << endl;
	svm_model * svm_model_test = test_model.getSvmModel();

	test_model.predict(test_node_datafile);

	test_trainer.crossValidate(test_model);
	//cout << "test" << endl;
	/*double calculate_accuracy = 0;
	double total_classifications = 0;
	double return_value = 0;
	vector<int> return_values;
	double test_accuracy = 0;
	int test_rows = test_node_datafile.getProblemLength();
	vector<double> y_test_labels = test_node_datafile.getYlabels();
	for (int row = 0; row < test_rows; row++)
	{
		return_value = svm_predict(svm_model_test, test_node[row]);
		return_values.push_back(return_value);
		if (return_value == y_test_labels[row]) ++test_accuracy;
		++total_classifications;
	}
	cout << "Return value of prediction = " << return_value << endl;
	double total_accuracy = 0;
	total_accuracy = test_accuracy / total_classifications;
	cout << "Accuracy = " << total_accuracy * 100 << "% (" << test_accuracy << "/" << total_classifications << ")" << endl;
	*/
	cout << "----------Done testing class!---------------" << endl;
	
	/* Reading training file*/
	

	cin.get();
    return 0;
}