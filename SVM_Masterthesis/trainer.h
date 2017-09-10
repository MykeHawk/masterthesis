#pragma once
#ifndef TRAINER_H
#define TRAINER_H

#include "svm.h"
#include "datafile.h"
#include "model.h"
#include <vector>
#include <algorithm>
#include <armadillo>

using namespace std;
using namespace arma;

struct boundaryValue {
	double lower_limit;
	double upper_limit;
	double stepsize;
};

class Trainer
{
public:
	Trainer();
	/* Training of a svm_model */
	void train(Model & model, DataFile & train, bool parameter_selection = false, bool feature_selection = false);
	/* Cross validation function*/
	double crossValidate(Model & model, int cross_validation_parameter, int show_nodes = 0);
	/* Standard grid search parameter selection */
	void parameterSelection(Model & model, DataFile & model_data);
	double parameterSelectionTrain(Model & model, DataFile & train, svm_parameter & custom_parameter);
	void checkOptimalParameters(double accuracy, svm_parameter training_parameter);
	void setBoundaryValue(double lower, double upper, double step, int boundary_parameter);
	void getBoundaryValue(double & lower, double & upper, double & step, int boundary_parameter);
	/* Grid optimisation in calculating optimal parameters between a wide range */
	void parameterGridOptimisation(Model & model, DataFile & train);
	double parameterGridOptimisationTrainer(Model & model, double c_value, double gamma_value, bool & optimal_accuracy_changed);
	int parameterGridOptimisationCheckAccuracy(vector<double> accuracy_values);
	void parameterGridOptimisationAssignValues(double & first, double & second, double & third, double & target_value);
	double calculateLog(double maximum, double minimum, double log_variable);
	/* Set the cross validation folds in the parameter optimisation and feature selection */
	int setCrossValidationFoldParameter(int custom_fold_parameter);
	int getCrossValidationFoldParameter();
	/* Feature selection */
	void featureSelection(DataFile & data_file);
	void forwardFeatureSelection(mat *data, mat *selectedFeatures, double targetAccuracy, bool usePCA = true, double varianceThreshold = 0);
	double calculateVariance(vec feature);
	void removeDuplicateFeatures(mat *data, unsigned int id, double margin);
	void pca(mat *data, mat *projection, vec *eigenvals);
	void scaleData(DataFile & data_file);
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

