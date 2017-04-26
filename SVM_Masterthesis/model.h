#pragma once
#ifndef MODEL_H
#define MODEL_H

#include "svm.h"
#include "datafile.h"

class Model
{
public:
	Model();
	~Model();
	/* Getters and setters */
	svm_model* getSvmModel(void);
	void setSvmModel(svm_model * generated_model);
	svm_parameter getParameter(void);
	void setParameter(svm_parameter user_parameter);
	DataFile getDatafile(void);
	void setDataFileTrainer(DataFile & trained);
	/* Functions */
	double predict(DataFile & testing_data);

private:
	svm_model* model;
	svm_parameter parameter;
	DataFile * trained_by;
};

#endif /* MODEL_H */
