#include "model.h"



Model::Model()
{
	/* Default parameters */
	parameter.svm_type = C_SVC;
	parameter.kernel_type = RBF;
	parameter.degree = 3;
	parameter.gamma = 0.5; //0.5
	parameter.coef0 = 0;
	parameter.nu = 0.5;
	parameter.cache_size = 100;
	parameter.eps = 1e-3; //0.001
	parameter.C = 1;
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

double Model::predict(DataFile & testing_data)
{
	double calculate_accuracy = 0;
	double total_classifications = 0;
	int return_value = 0;
	vector<int> return_values;
	int test_rows = testing_data.getProblemLength();
	vector<double> y_test_labels = testing_data.getYlabels();
	//svm_node** testing_node = testing_data.getNode();
	for (int row = 0; row < test_rows; row++)
	{
		return_value = svm_predict(model, testing_data.getNode()[row]);
		return_values.push_back(return_value);
		if (return_value == y_test_labels[row]) ++calculate_accuracy;
		++total_classifications;
	}
	double total_accuracy = calculate_accuracy / total_classifications * 100;
	cout << "Return value of prediction = " << return_value << endl;
	cout << "Accuracy = " << total_accuracy << "% (" << calculate_accuracy << "/" << total_classifications << ")" << endl;

	std::ofstream output_file("./return_values_class.txt");
	std::ostream_iterator<int> output_iterator(output_file, "\n");
	std::copy(return_values.begin(), return_values.end(), output_iterator);

	return total_accuracy;
}


Model::~Model()
{
}
