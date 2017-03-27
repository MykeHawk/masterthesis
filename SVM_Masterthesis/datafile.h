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
	DataFile(string path);
	~DataFile();
	svm_problem & getProblem(void);
	svm_node** getNode(void);
	int getProblemLength(void);
	vector<double> getYlabels(void);
	//vector<int> &getValueNodes(void) const;
	int processFile(ifstream& myfile);
	int generateProblem(void);
private:
	svm_node** svm_node_datafile;
	svm_problem  svm_problem_datafile;
	vector<double> y_labels;
	vector<int> index_nodes;
	vector<int> value_nodes;
	int problem_length = 0;
};

#endif /* DATAFILE_H */