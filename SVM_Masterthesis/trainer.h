#pragma once
#ifndef TRAINER_H
#define TRAINER_H

#include "svm.h"
#include "datafile.h"
#include "model.h"

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
	void crossValidate(Model & model, int cross_validation_parameter);
	void parameterSelection(Model & model, DataFile & train, DataFile & testing_file);
	double parameterSelectionTrain(Model & model, DataFile & train, DataFile & testing_file, svm_parameter & custom_parameter);
	void setBoundaryValue(double lower, double upper, double step, int boundary_parameter);
	~Trainer();
private:
	boundaryValue c_value;
	boundaryValue gamma_value;
};

#endif /* TRAINER_H */

