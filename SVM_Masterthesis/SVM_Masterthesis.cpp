// SVM_Masterthesis.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <iterator>
#include <fstream>
#include <stdio.h>
//#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
//#include <crtdbg.h>
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
	
	/* Testing training and prediction*/
	test_trainer.train(test_model, training_file);
	DataFile test_model_datafile = test_model.getDatafile();
	svm_node ** test_datafile_node = test_model_datafile.getNode();
	cout << "++testing datafile node from trained model: " << test_datafile_node[27][0].index << endl;
	svm_model * svm_model_test = test_model.getSvmModel();
	test_model.predict(test_node_datafile);
	cout << "----------Done testing training and prediction!---------------" << endl;
	cout << endl;

	/* Testing user selection of parameters */
	svm_parameter parameter;
	parameter.svm_type = C_SVC;
	parameter.kernel_type = RBF;
	parameter.degree = 3;
	parameter.gamma = 0; //0.5
	parameter.coef0 = 0;
	parameter.nu = 0.5;
	parameter.cache_size = 100;
	parameter.eps = 1e-3; //0.001
	parameter.C = 1;
	parameter.p = 0.1;
	parameter.nr_weight = 0;
	parameter.weight_label = NULL;
	parameter.weight = NULL;
	parameter.probability = 0;
	parameter.shrinking = 1;
	test_model.setParameter(parameter);
	test_trainer.train(test_model, training_file);
	test_model.predict(test_node_datafile);
	cout << "----------Done testing user custom parameters!---------------" << endl;
	cout << endl;

	/* Testing parameter optimisation */
	test_trainer.parameterSelection(test_model, training_file, test_node_datafile);
	cout << "----------Done testing standard parameter optimisation!---------------" << endl;
	cout << endl;
	test_trainer.setBoundaryValue(0.1, 0.5, 0.1, 0);
	//test_trainer.parameterSelection(test_model, training_file, test_node_datafile);

	/* Testing crossvalidation */
	test_trainer.crossValidate(test_model, 5);


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
	
//	_CrtDumpMemoryLeaks();
	cin.get();
    return 0;
}