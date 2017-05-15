#pragma once
#ifndef TRAINER_H
#define TRAINER_H

#include "svm.h"
#include "datafile.h"
#include "model.h"
#include <vector>
#include <algorithm>

struct boundaryValue {
	double lower_limit;
	double upper_limit;
	double stepsize;
};

class Trainer
{
public:
	Trainer();
	void train(Model & model, DataFile & train, bool parameter_selection = false, bool feature_selection = false);
	double crossValidate(Model & model, int cross_validation_parameter, int show_nodes = 0);
	void parameterSelection(Model & model, DataFile & train, DataFile & testing_file);
	double parameterSelectionTrain(Model & model, DataFile & train, DataFile & testing_file, svm_parameter & custom_parameter);
	void checkOptimalParameters(double accuracy, svm_parameter training_parameter);
	void setBoundaryValue(double lower, double upper, double step, int boundary_parameter);
	void getBoundaryValue(double & lower, double & upper, double & step, int boundary_parameter);
	void parameterGridOptimisation(Model & model, DataFile & train);
	double parameterGridOptimisationTrainer(Model & model, double c_value, double gamma_value);
	int parameterGridOptimisationCheckAccuray(vector<double> accuracy_values);
	void parameterGridOptimisationAssignValues(double & first, double & second, double & third, double & target_value);
	int setCrossValidationFoldParameter(int custom_fold_parameter);
	int getCrossValidationFoldParameter();
	~Trainer();
private:
	boundaryValue c_value;
	boundaryValue gamma_value;
	double optimal_accuracy = 0;
	double optimal_C_value = 0;
	double optimal_gamma_value = 0;
	int cross_validation_fold_parameter = 5;
};

#endif /* TRAINER_H */

