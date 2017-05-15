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
	cout << "++testing datafile node from trained model: " << test_datafile_node[0][0].index << endl;
	svm_model * svm_model_test = test_model.getSvmModel();
	test_model.predict(test_node_datafile);
	cout << "----------Done testing training and prediction!---------------" << endl;
	cout << endl;


	/* Testing user selection of parameters */
	/*
	svm_parameter parameter;
	parameter.svm_type = C_SVC;
	parameter.kernel_type = RBF;
	parameter.degree = 3;
	parameter.gamma = 10; //0.5
	parameter.coef0 = 0;
	parameter.nu = 0.5;
	parameter.cache_size = 100;
	parameter.eps = 1e-3; //0.001
	parameter.C = 1000;
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
	*/
	/* Testing parameter optimisation */
	/*
	test_trainer.parameterSelection(test_model, training_file, test_node_datafile);
	cout << "----------Done testing standard parameter optimisation!---------------" << endl;
	cout << endl;
	double boundary_upper, boundary_lower, boundary_step;
	test_trainer.getBoundaryValue(boundary_lower, boundary_upper, boundary_step, 0);
	cout << boundary_lower << " " << boundary_upper << " " << boundary_step << endl;
	test_trainer.setBoundaryValue(0, 0.5, 0.1, 0);
	test_trainer.getBoundaryValue(boundary_lower, boundary_upper, boundary_step, 0);
	cout << boundary_lower << " " << boundary_upper << " " << boundary_step << endl;
	test_trainer.setCrossValidationFoldParameter(2);
	test_trainer.parameterSelection(test_model, training_file, test_node_datafile);
	cout << "----------Done testing custom parameter optimisation!---------------" << endl;
	cout << endl;
	*/
	/* Testing grid optimisation search*/
	
	test_trainer.parameterGridOptimisation(test_model, training_file);
	cout << "----------Done testing parameter optimisation with grid !---------------" << endl;
	
	/* Testing crossvalidation */
	/*
	double accuracy = test_trainer.crossValidate(test_model, 5);
	cout << "Crossvalidation accuracy equals: " << accuracy << endl;
	cout << "----------Done testing cross validation!---------------" << endl;
	cout << endl;
	*/

	cout << "----------Done testing class!---------------" << endl;
	
	cin.get();
    return 0;
}