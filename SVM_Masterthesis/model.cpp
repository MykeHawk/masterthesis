//#define _CRTDBG_MAP_ALLOC
//#include <crtdbg.h>
//#include <stdlib.h>
#include "model.h"

Model::Model()
{
	/* Default parameters */
	parameter.svm_type = C_SVC;
	parameter.kernel_type = RBF;
	parameter.degree = 3;
	parameter.gamma = 0.05; //0.5
	parameter.coef0 = 0;
	parameter.nu = 0.5;
	parameter.cache_size = 100;
	parameter.eps = 1e-3; //0.001
	parameter.C = 5;
	parameter.p = 0.1;
	parameter.nr_weight = 0;
	parameter.weight_label = NULL;
	parameter.weight = NULL;
	parameter.probability = 0;
	parameter.shrinking = 1;
}

svm_model * Model::getSvmModel(void)
{
	return model;
}

void Model::setSvmModel(svm_model * generated_model)
{
	model = generated_model;
}

svm_parameter Model::getParameter(void)
{
	return parameter;
}

void Model::setParameter(svm_parameter user_parameter)
{
	parameter = user_parameter;
}

DataFile Model::getDatafile(void)
{
	return *trained_by;
}

void Model::setDataFileTrainer(DataFile & trained)
{
	trained_by = &trained;
}

vector<double> Model::getSvmPrediction()
{
	return svm_prediction;
}

double Model::predict(DataFile & testing_data, bool normalize_data)
{
	//svm_node ** node_data = testing_data.getNode();
	double calculate_accuracy = 0;
	double total_classifications = 0;
	double return_value = 0;
	svm_prediction.clear();
	//vector<double> return_values;
	int test_rows = testing_data.getProblemLength();
	svm_node ** test_data = testing_data.getNode();
	if (normalize_data && trained_by->getDataRange() != 0) testDataScale(test_data, test_rows);
	vector<double> y_test_labels = testing_data.getYlabels();
	//svm_node** testing_node = testing_data.getNode();
	for (int row = 0; row < test_rows; row++)
	{
		return_value = svm_predict(model, testing_data.getNode()[row]);
		svm_prediction.push_back(return_value);
		if (!y_test_labels.empty())
		{
			if (return_value == y_test_labels[row]) ++calculate_accuracy;
			++total_classifications;
		}
		
	}

	std::ofstream output_file("./return_values_class.txt");
	std::ostream_iterator<double> output_iterator(output_file, "\n");
	std::copy(svm_prediction.begin(), svm_prediction.end(), output_iterator);
	if (!y_test_labels.empty())
	{
		double total_accuracy = calculate_accuracy / total_classifications * 100;
		cout << "Return value of prediction = " << return_value << endl;
		cout << "Accuracy = " << total_accuracy << "% (" << calculate_accuracy << "/" << total_classifications << ")" << endl;
		return total_accuracy;
	}

	return return_value;
}

double Model::predictNode(svm_node ** custom_node, int node_row_length)
{
	double calculate_accuracy = 0;
	double total_classifications = 0;
	double return_value = 0;
	svm_prediction.clear();
	for (int row = 0; row < node_row_length; row++)
	{
		return_value = svm_predict(model, custom_node[row]);
		svm_prediction.push_back(return_value);
	}

	std::ofstream output_file("./return_values_class.txt");
	std::ostream_iterator<double> output_iterator(output_file, "\n");
	std::copy(svm_prediction.begin(), svm_prediction.end(), output_iterator);

	return return_value;
}

void Model::testDataScale(svm_node ** data_node, int node_length)
{
	int column = 0;
	double minimum = trained_by->getDataMinimum();
	double range = trained_by->getDataRange();
	double maximum = minimum + range;
	//cout << "Testing range and minimum: " << range << " " << minimum << " " << maximum << endl;
	for (int i = 0; i < node_length; i++)
	{
		column = 0;
		while (data_node[i][column].index != -1)
		{
			if (range != minimum)
			{
				//cout << data_node[i][column].value << " ";
				data_node[i][column].value = (data_node[i][column].value - minimum) / range;
				//cout << data_node[i][column].value << " ";
				if (data_node[i][column].value > 1)
				{
					data_node[i][column].value = 1;
				}
				else if (data_node[i][column].value < 0)
				{
					data_node[i][column].value = 0;
				}
			}
			//else data_node[i][column].value = data_node[i][column].value / range;
			++column;
		}
		//cout << endl;
	}
}



Model::~Model()
{
	//svm_destroy_param(&parameter);
	//delete parameter;
	//svm_free_and_destroy_model(&model);
}
