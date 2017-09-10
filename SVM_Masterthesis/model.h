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
	svm_model* getSvmModel(void);
	void setSvmModel(svm_model * generated_model);
	svm_parameter getParameter(void);
	void setParameter(svm_parameter user_parameter);
	DataFile getDatafile(void);
	void setDataFileTrainer(DataFile & trained);
	vector<double> getSvmPrediction();
	double predict(DataFile & testing_data, bool normalize_data = true);
	double predictNode(svm_node** custom_node, int node_row_length); //performing a prediction with just 1 node
	void testDataScale(svm_node ** data_node, int node_length);
private:
	vector<double> svm_prediction;
	svm_model* model;
	svm_parameter parameter;
	DataFile * trained_by;
	double range_of_data;
	double data_minimum;
};

#endif /* MODEL_H */
