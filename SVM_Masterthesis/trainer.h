#pragma once
#ifndef TRAINER_H
#define TRAINER_H

#include "svm.h"
#include "datafile.h"
#include "model.h"
class Trainer
{
public:
	Trainer();
	void train(Model & model, DataFile & train, bool parameter_selection = false, bool feature_selection = false);
	void crossValidate(Model & model);
	~Trainer();
};

#endif /* TRAINER_H */

