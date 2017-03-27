#include "trainer.h"



Trainer::Trainer()
{
}

void Trainer::train(Model & model, DataFile & train, bool parameter_selection, bool feature_selection)
{
	svm_problem training_problem = train.getProblem();
	svm_parameter training_parameter = model.getParameter();
	svm_model * train_model = svm_train(&training_problem, &training_parameter);
	model.setDataFileTrainer(train);
	model.setSvmModel(train_model);

}

void Trainer::crossValidate(Model & model)
{
	int k_fold_parameter = 2;
	//Model test_model;
	DataFile datafile_training = model.getDatafile();
	svm_node ** node_datafile_training = datafile_training.getNode();
	svm_model * model_training = model.getSvmModel();
	vector<double> datafile_traininglabels = datafile_training.getYlabels();
	vector<double> crossvalidate_labels;

	double return_value = 0;
	int crossvalidate_maximum = datafile_training.getProblemLength();
	int number_of_iterations = 0;
	//int crossvalidate_iterations = 0;
	int crossvalidate_parameter = (crossvalidate_maximum / k_fold_parameter) ;
	cout << "Number of data per folds: " << crossvalidate_parameter << endl;
	cout << "Problem length: " << crossvalidate_maximum << endl;
	svm_node ** crossvalidate_node = new svm_node*[crossvalidate_parameter];
	//cout << "test maximum of problem in cross validation: " << node_datafile_training[crossvalidate_maximum - 1][0].index << endl;

	vector<svm_node**> crossvalidate_testing_bin_nodes;
	vector<svm_node**> crossvalidate_training_bin_nodes;
	svm_node ** crossvalidate_testing_bin_node = new svm_node*[crossvalidate_parameter];
	svm_node ** crossvalidate_training_bin_node = new svm_node*[crossvalidate_maximum];
	int bin = 0;
	int total_iterations = 0;
	int crossvalidate_iterations = 0;
	int algorithm_iterator_min = 0;
	int algorithm_iterator_max = 0;
	while (bin < k_fold_parameter)
	{
		cout << "-- bin -- = " << bin << endl;
		algorithm_iterator_max = (bin + 1) * (crossvalidate_parameter);
		for(int i = 0; i < crossvalidate_maximum; i++)
		{
			if (i >= algorithm_iterator_min && i < algorithm_iterator_max)
			{
				//cout << "-- putting in testing data i = " << i << "--" << endl;
				crossvalidate_testing_bin_node[i] = node_datafile_training[i];
			}
			else
			{
				//cout << "putting in training data i = " << i << endl;
				crossvalidate_training_bin_node[i] = node_datafile_training[i];
			}
		} 

		//cout << "--->crossvalidate testing: " << crossvalidate_testing_bin_node[0][0].index << endl;
		
		for (int j = 0; j < crossvalidate_maximum; j++)
		{
			if (j >= algorithm_iterator_min && j < algorithm_iterator_max)
			{
				//cout << "-- putting in testing data i = " << i << "--" << endl;
				cout << j << ") " << "--->crossvalidate testing: " << crossvalidate_testing_bin_node[j][0].index << endl;
			}
			else
			{
				//cout << "putting in training data i = " << i << endl;
				cout << j << ") " << "crossvalidate training: " << crossvalidate_training_bin_node[j][0].index << endl;
			}
		}
		++bin;
		algorithm_iterator_min = bin * (crossvalidate_parameter);
		crossvalidate_testing_bin_nodes.push_back(crossvalidate_testing_bin_node);
		crossvalidate_training_bin_nodes.push_back(crossvalidate_training_bin_node);
		//svm_node ** crossvalidate_testing_bin_node = new svm_node*[crossvalidate_parameter];
		//svm_node ** crossvalidate_training_bin_node = new svm_node*[crossvalidate_maximum];
	}




	/*
	while (number_of_iterations < crossvalidate_maximum)
	{
		//cout << "Data from cross validation: " << node_datafile_training[number_of_iterations][0].index << " " << number_of_iterations << endl;
		cout << "crossvalidate iteration: " << crossvalidate_iterations << endl;
		crossvalidate_node[crossvalidate_iterations] = node_datafile_training[number_of_iterations];
		// TO DO FIX CROSS VALIDATION LABELING
		//crossvalidate_labels
		cout << "values from nodes on cross: " << crossvalidate_node[crossvalidate_iterations][0].index << endl;
		++number_of_iterations;
		++crossvalidate_iterations;
		if (crossvalidate_iterations == crossvalidate_parameter || number_of_iterations == crossvalidate_maximum)
		{
			double calculate_accuracy = 0;
			double total_classifications = 0;
			for (int row = 0; row < crossvalidate_iterations; row++)
			{
				return_value = svm_predict(model_training, crossvalidate_node[row]);
				cout << "Data file training lables: " << datafile_traininglabels[row] << endl;
				//return_values.push_back(return_value);
				if (return_value == datafile_traininglabels[row]) ++calculate_accuracy;
				++total_classifications;
				cout << return_value << endl;
			}
			cout << "Accuracy from cross validation = " << calculate_accuracy / total_classifications * 100 << "% (" << calculate_accuracy << "/" << total_classifications << ")" << endl;
			crossvalidate_iterations = 0;
		}
			
	}
	*/
	/*for (int i = 0; i < 5; i++)
	{
		cout << "Data from cross validation: " << node_datafile_training[i][0].index << endl;
	}*/



}


Trainer::~Trainer()
{
}
