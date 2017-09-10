#pragma once
#ifndef DATAFILE_H
#define DATAFILE_H

/* Include files */
#include "svm.h"
#include <memory>
#include <iterator>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
using namespace std;

class DataFile
{
public:
	DataFile();
	DataFile(string path);
	~DataFile();
	svm_problem & getProblem(void);
	svm_node** getNode(void);
	void setNode(svm_node** custom_node);
	int getProblemLength(void);
	void setProblemLength(int custom_problem_length);
	vector<double> getYlabels(void);
	vector<double> getIndexNodes(void);
	vector<double> getValueNodes(void);
	void setYLabels(vector<double> custom_y_labels);
	double getDataRange();
	void setDataRange(double range);
	double getDataMinimum();
	void setDataMinimum(double minimum);
	//vector<int> &getValueNodes(void) const;
	int processFile(ifstream& myfile);
	int processFileNoLabels(string path);
	int generateProblem(void);
	int generateProblemFromNodes(void);
private:
	svm_node** svm_node_datafile;
	svm_problem  svm_problem_datafile;
	vector<double> y_labels;
	vector<double> index_nodes;
	vector<double> value_nodes;
	int problem_length = 0;
	double data_range_minimum_maximum = 0;
	double data_minimum_value = 0;
};

#endif /* DATAFILE_H */