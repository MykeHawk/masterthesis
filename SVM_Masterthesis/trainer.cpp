#include "trainer.h"

void print_null(const char *s) {}
void(*print_func)(const char*) = NULL;

Trainer::Trainer()
{
	// Init parameter selection values
	c_value.lower_limit = 0.5;
	c_value.upper_limit = 2;
	c_value.stepsize = 0.5;

	gamma_value.lower_limit = 0.2;
	gamma_value.upper_limit = 1;
	gamma_value.stepsize = 0.2;

	/* Disable libsvm output print*/
	print_func = &print_null;
	svm_set_print_string_function(print_func);

}

void Trainer::train(Model & model, DataFile & train, bool parameter_selection, bool feature_selection)
{
	svm_problem training_problem = train.getProblem();
	svm_parameter training_parameter = model.getParameter();

	svm_model * train_model = svm_train(&training_problem, &training_parameter);
	model.setDataFileTrainer(train);
	model.setSvmModel(train_model);

}

double Trainer::crossValidate(Model & model, int cross_validation_parameter, int show_nodes)
{
	// Test parameter, needs to be given as parameter to function
	int k_fold_parameter = cross_validation_parameter;

	/* Get data from the model  */
	DataFile datafile_training = model.getDatafile();
	svm_node ** node_datafile_training = datafile_training.getNode();
	//svm_model * model_training = model.getSvmModel();
	svm_parameter model_parameter = model.getParameter();
	vector<double> datafile_traininglabels = datafile_training.getYlabels();

	/* Setting up cross validation parameters */
	int crossvalidate_maximum = datafile_training.getProblemLength();
	int crossvalidate_parameter = (crossvalidate_maximum / k_fold_parameter);
	cout << "Number of data per folds: " << crossvalidate_parameter << endl;
	cout << "Problem length: " << crossvalidate_maximum << endl;

	/* Initialising variables */
	svm_node *** crossvalidate_testing_bin_nodes = new svm_node**[k_fold_parameter];
	svm_node *** crossvalidate_training_bin_nodes = new svm_node**[k_fold_parameter];
	int bin = 0;
	int total_iterations = 0;
	int crossvalidate_iterations = 0;
	int algorithm_iterator_min = 0;
	int algorithm_iterator_max = 0;
	int number_of_trainings = crossvalidate_maximum - crossvalidate_parameter;
	double total_classifications = 0;
	double return_value = 0;
	double total_accuracy = 0;

	/* Start of cross validation depending on the number of folds */
	while (bin < k_fold_parameter)
	{
		int testing_row = 0;
		int training_row = 0;
		crossvalidate_testing_bin_nodes[bin] = new svm_node*[crossvalidate_parameter];
		crossvalidate_training_bin_nodes[bin] = new svm_node*[number_of_trainings];
		svm_problem crossvalidation_svm_problem;
		crossvalidation_svm_problem.l = number_of_trainings;
		crossvalidation_svm_problem.y = new double[number_of_trainings];
		vector<double> testing_labels;
		//cout << "-- bin -- = " << bin << endl;
		algorithm_iterator_max = (bin + 1) * (crossvalidate_parameter);
		for (int i = 0; i < crossvalidate_maximum; i++)
		{
			if (i >= algorithm_iterator_min && i < algorithm_iterator_max)
			{
				crossvalidate_testing_bin_nodes[bin][testing_row] = node_datafile_training[i];
				testing_labels.push_back(datafile_traininglabels[i]);
				++testing_row;
			}
			else
			{
				crossvalidate_training_bin_nodes[bin][training_row] = node_datafile_training[i];
				crossvalidation_svm_problem.y[training_row] = datafile_traininglabels[i];
				++training_row;
			}
		}

		crossvalidation_svm_problem.x = crossvalidate_training_bin_nodes[bin];
		svm_model * crossvalidation_model = svm_train(&crossvalidation_svm_problem, &model_parameter);
		for (int m = 0; m < testing_row; m++)
		{
			return_value = svm_predict(crossvalidation_model, crossvalidate_testing_bin_nodes[bin][m]);
			if (return_value == testing_labels[m]) ++total_accuracy;
			++total_classifications;
		}
		//total_accuracy = 0;
		//total_classifications = 0;
		//cout << "Bin accuracy: " << total_accuracy / total_classifications * 100 << endl;
		++bin;
		algorithm_iterator_min = bin * (crossvalidate_parameter);
		//cout << "value of Y at 0:  " << crossvalidation_svm_problem.y[0] << endl;
	}
	double crossvalidation_accuracy = total_accuracy / total_classifications * 100;
	//cout << "crossvalidation accuracy equals: " << crossvalidation_accuracy << endl;
	//Testing if data is correct
	//int number_of_trainings = crossvalidate_maximum - crossvalidate_parameter;
	if (show_nodes == 1)
	{
		for (int j = 0; j < bin; j++)
		{
			cout << "---- Showing first index of each node on bin: " << j << " ----" << endl;
			cout << "-- Printing testing nodes --" << endl;
			for (int k = 0; k < crossvalidate_parameter; k++)
			{
				cout << "Testing node first index on position " << k << " equals: " << crossvalidate_testing_bin_nodes[j][k][0].index << " with value: " << crossvalidate_testing_bin_nodes[j][k][0].value << endl;
			}
			cout << endl;
			cout << "-- Printing training nodes --" << endl;
			for (int l = 0; l < number_of_trainings; l++)
			{
				cout << "Training node first index on position " << l << " equals: " << crossvalidate_training_bin_nodes[j][l][0].index << " with value: " << crossvalidate_training_bin_nodes[j][l][0].value << endl;
			}
		}
	}
	return crossvalidation_accuracy;

}

void Trainer::parameterSelection(Model & model, DataFile & train, DataFile & testing_file)
{
	svm_problem train_problem = train.getProblem();
	svm_parameter training_parameter = model.getParameter();

	training_parameter.C = c_value.lower_limit;
	training_parameter.gamma = gamma_value.lower_limit;
	double accuracy = 0;
	optimal_accuracy = 0;

	while (1)
	{
		if (accuracy >= 99)
		{
			training_parameter.C = optimal_C_value;
			training_parameter.gamma = optimal_gamma_value;
			model.setParameter(training_parameter);
			cout << "Accuracy is near perfect, stopping the parameter selection" << endl;
			cout << "Automatically adjusting values of C and gamma to: " << optimal_C_value << " " << optimal_gamma_value << endl;
			break;
		}
		cout << "----training with C:" << training_parameter.C << " and gamma:" << training_parameter.gamma << "-----" << endl;
		accuracy = Trainer::parameterSelectionTrain(model, train, testing_file, training_parameter);
		cout << "Accuracy: " << accuracy << endl;
		Trainer::checkOptimalParameters(accuracy, training_parameter);
		training_parameter.C += c_value.stepsize;

		if (training_parameter.C == c_value.upper_limit && training_parameter.gamma == gamma_value.upper_limit)
		{
			cout << "----training with C:" << training_parameter.C << " and gamma:" << training_parameter.gamma << "----" <<endl;
			accuracy = Trainer::parameterSelectionTrain(model, train, testing_file, training_parameter);
			cout << "Accuracy: " << accuracy << endl;
			Trainer::checkOptimalParameters(accuracy, training_parameter);
			training_parameter.C = optimal_C_value;
			training_parameter.gamma = optimal_gamma_value;
			model.setParameter(training_parameter);
			cout << "Automatically adjusting values of C and gamma to: " << optimal_C_value << " " << optimal_gamma_value << endl;
			cout << "End!" << endl;
			break;
		}

		if (training_parameter.C == c_value.upper_limit && accuracy <= 99)
		{
			cout << "----training with C:" << training_parameter.C << " and gamma:" << training_parameter.gamma << "----" << endl;
			accuracy = Trainer::parameterSelectionTrain(model, train, testing_file, training_parameter);
			cout << "Accuracy: " << accuracy << endl;
			Trainer::checkOptimalParameters(accuracy, training_parameter);
			training_parameter.C = c_value.lower_limit;
			training_parameter.gamma += gamma_value.stepsize;
		}

	}

}

double Trainer::parameterSelectionTrain(Model & model, DataFile & train, DataFile & testing_file, svm_parameter & custom_parameter)
{
	model.setParameter(custom_parameter);
	Trainer::train(model, train);
	double accuracy = Trainer::crossValidate(model, cross_validation_fold_parameter);
	return accuracy;
}

void Trainer::checkOptimalParameters(double accuracy, svm_parameter training_parameter)
{
	if (optimal_accuracy == 0 || optimal_accuracy < accuracy)
	{
		optimal_accuracy = accuracy;
		optimal_C_value = training_parameter.C;
		optimal_gamma_value = training_parameter.gamma;
		cout << "New optimal accuracy: " << optimal_accuracy << " with C and Gamma: " << optimal_C_value << " " << optimal_gamma_value << endl;
	}
}

void Trainer::setBoundaryValue(double lower, double upper, double step, int boundary_parameter)
{
	if (boundary_parameter == 0)
	{
		c_value.lower_limit = lower;
		c_value.upper_limit = upper;
		c_value.stepsize = step;
	}
	else if (boundary_parameter == 1)
	{
		gamma_value.lower_limit = lower;
		gamma_value.upper_limit = upper;
		gamma_value.stepsize = step;
	}
}

void Trainer::getBoundaryValue(double & lower, double & upper, double & step, int boundary_parameter)
{
	if (boundary_parameter == 0)
	{
		lower = c_value.lower_limit;
		upper = c_value.upper_limit;
		step = c_value.stepsize;
	}
	else if (boundary_parameter == 1)
	{
		lower = gamma_value.lower_limit;
		upper = gamma_value.upper_limit;
		step = gamma_value.stepsize;
	}
}

void Trainer::parameterGridOptimisation(Model & model, DataFile & train)
{
	// Using 4 square method
	svm_problem train_problem = train.getProblem();
	svm_parameter training_parameter = model.getParameter();
	optimal_accuracy = 0;

	double minimum_c_value = 0.01;
	//double maximum_c_value = 100000000000;
	double maximum_c_value = 10000;
	double minimum_gamma_value = 0.001;
	double maximum_gamma_value = 100;

	double initial_c_value = (minimum_c_value + maximum_c_value) / 2;
	double initial_gamma_value = (minimum_gamma_value + maximum_gamma_value) / 2;
	training_parameter.C = initial_c_value;
	training_parameter.gamma = initial_gamma_value;

	//cross_validation_fold_parameter = 2;
	model.setParameter(training_parameter);
	double accuracy = Trainer::crossValidate(model, 150);
	cout << "Accuracy: " << accuracy << endl;
	optimal_accuracy = accuracy;
	double test_accuracy = parameterGridOptimisationTrainer(model, initial_c_value, initial_gamma_value);
	cout << "test accuracy: " << test_accuracy << endl;

	double first_square_c_value = initial_c_value;
	double first_square_gamma_value = initial_gamma_value;

	double second_square_c_value = initial_c_value;
	double second_square_gamma_value = initial_gamma_value;

	double third_square_c_value = initial_c_value;
	double third_square_gamma_value = initial_gamma_value;

	double fourth_square_c_value = initial_c_value;
	double fourth_square_gamma_value = initial_gamma_value;

	vector<double> squares_accuracy_vector;
	for (int i = 0; i < 15; i++)
	{
		first_square_c_value = (first_square_c_value + minimum_c_value) / 2;
		first_square_gamma_value = (first_square_gamma_value + maximum_gamma_value) / 2;
		cout << "Values of C and gamma for 1: " << first_square_c_value << " " << first_square_gamma_value << endl;
		accuracy = parameterGridOptimisationTrainer(model, first_square_c_value, first_square_gamma_value);
		cout << "Accuracy: " << accuracy << endl;
		squares_accuracy_vector.push_back(accuracy);

		second_square_c_value = (second_square_c_value + maximum_c_value) / 2;
		second_square_gamma_value = (second_square_gamma_value + maximum_gamma_value) / 2;
		cout << "Values of C and gamma for 2: " << second_square_c_value << " " << second_square_gamma_value << endl;
		accuracy = parameterGridOptimisationTrainer(model, second_square_c_value, second_square_gamma_value);
		cout << "Accuracy: " << accuracy << endl;
		squares_accuracy_vector.push_back(accuracy);

		third_square_c_value = (third_square_c_value + maximum_c_value) / 2;
		third_square_gamma_value = (third_square_gamma_value + minimum_gamma_value) / 2;
		cout << "Values of C and gamma for 3: " << third_square_c_value << " " << third_square_gamma_value << endl;
		accuracy = parameterGridOptimisationTrainer(model, third_square_c_value, third_square_gamma_value);
		cout << "Accuracy: " << accuracy << endl;
		squares_accuracy_vector.push_back(accuracy);

		fourth_square_c_value = (fourth_square_c_value + minimum_c_value) / 2;
		fourth_square_gamma_value = (fourth_square_gamma_value + minimum_gamma_value) / 2;
		cout << "Values of C and gamma for 4: " << fourth_square_c_value << " " << fourth_square_gamma_value << endl;
		accuracy = parameterGridOptimisationTrainer(model, fourth_square_c_value, fourth_square_gamma_value);
		cout << "Accuracy: " << accuracy << endl;
		squares_accuracy_vector.push_back(accuracy);
		int best_accuracy_square = parameterGridOptimisationCheckAccuray(squares_accuracy_vector);
		cout << "Best square: " << best_accuracy_square << endl;
		if (best_accuracy_square == 0)
		{
			parameterGridOptimisationAssignValues(second_square_c_value, third_square_c_value, fourth_square_c_value, first_square_c_value);
			parameterGridOptimisationAssignValues(second_square_gamma_value, third_square_gamma_value, fourth_square_gamma_value, first_square_gamma_value);
			maximum_c_value = (first_square_c_value * 2) - minimum_c_value;
			minimum_gamma_value = (first_square_gamma_value * 2) - maximum_gamma_value;
		}
		else if (best_accuracy_square == 1)
		{
			parameterGridOptimisationAssignValues(first_square_c_value, third_square_c_value, fourth_square_c_value, second_square_c_value);
			parameterGridOptimisationAssignValues(first_square_gamma_value, third_square_gamma_value, fourth_square_gamma_value, second_square_gamma_value);
			minimum_c_value = (second_square_c_value * 2) - maximum_c_value;
			minimum_gamma_value = (second_square_gamma_value * 2) - maximum_gamma_value;
		}
		else if (best_accuracy_square == 2)
		{
			parameterGridOptimisationAssignValues(first_square_c_value, second_square_c_value, fourth_square_c_value, third_square_c_value);
			parameterGridOptimisationAssignValues(first_square_gamma_value, second_square_gamma_value, fourth_square_gamma_value, third_square_gamma_value);
			minimum_c_value = (third_square_c_value * 2) - maximum_c_value;
			maximum_gamma_value = (third_square_gamma_value * 2) - minimum_gamma_value;
		}
		else
		{
			parameterGridOptimisationAssignValues(first_square_c_value, second_square_c_value, third_square_c_value, fourth_square_c_value);
			parameterGridOptimisationAssignValues(first_square_gamma_value, second_square_gamma_value, third_square_gamma_value, fourth_square_gamma_value);
			maximum_c_value = (fourth_square_c_value * 2) - minimum_c_value;
			maximum_gamma_value = (fourth_square_gamma_value * 2) - minimum_gamma_value;
		
		}
		
	}


}

double Trainer::parameterGridOptimisationTrainer(Model & model, double c_value, double gamma_value)
{
	svm_parameter training_parameter = model.getParameter();
	training_parameter.C = c_value;
	training_parameter.gamma = gamma_value;
	model.setParameter(training_parameter);
	double accuracy = Trainer::crossValidate(model, 150);

	return accuracy;
}

int Trainer::parameterGridOptimisationCheckAccuray(vector<double> accuracy_values)
{
	int square = 0;
	auto result = minmax_element(accuracy_values.begin(), accuracy_values.end());
	int minimum = result.first - accuracy_values.begin();
	int maximum = result.second - accuracy_values.begin();
	cout << "Minimum and maximum: " << minimum << " " << maximum << endl;
	if (accuracy_values[minimum] == accuracy_values[maximum]) return 3;
	return maximum;
}

void Trainer::parameterGridOptimisationAssignValues(double & first, double & second, double & third, double & target_value)
{
	first = target_value;
	second = target_value;
	third = target_value;
}

int Trainer::setCrossValidationFoldParameter(int custom_fold_parameter)
{
	cross_validation_fold_parameter = custom_fold_parameter;
	return cross_validation_fold_parameter;
}

int Trainer::getCrossValidationFoldParameter()
{
	return cross_validation_fold_parameter;
}


Trainer::~Trainer()
{
}
