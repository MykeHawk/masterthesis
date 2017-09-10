#include "trainer.h"

/* Disabling libsvm output print */
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
	svm_parameter model_parameter = model.getParameter();
	vector<double> datafile_traininglabels = datafile_training.getYlabels();

	/* Setting up cross validation parameters */
	int crossvalidate_maximum = datafile_training.getProblemLength();
	int crossvalidate_parameter = (crossvalidate_maximum / k_fold_parameter);
	if (crossvalidate_parameter == 0) crossvalidate_parameter = 1;
	cout << "Number of data per folds: " << crossvalidate_parameter << endl;
	cout << "Problem length: " << crossvalidate_maximum << endl;

	/* Initialising variables */
	svm_node *** crossvalidate_testing_bin_nodes = new svm_node**[k_fold_parameter];
	svm_node *** crossvalidate_training_bin_nodes = new svm_node**[k_fold_parameter];
	int bin = 0; //number of bins according to the k-fold parameter
	int algorithm_iterator_min = 0; //determine start of testdata
	int algorithm_iterator_max = 0; //determine end of testdata
	int number_of_trainings = crossvalidate_maximum - crossvalidate_parameter; //amount of trainingsdata in crossvalidation
	double total_classifications = 0; //parameter used to calculate accuracy
	double return_value = 0; //parameter used to calculate accuracy
	double total_accuracy = 0; //parameter used to calculate accuracy

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
		++bin;
		algorithm_iterator_min = bin * (crossvalidate_parameter);
	}
	double crossvalidation_accuracy = total_accuracy / total_classifications * 100;

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

void Trainer::parameterSelection(Model & model, DataFile & model_data)
{
	//DataFile model_data = model.getDatafile();
	svm_problem train_problem = model_data.getProblem();
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
		accuracy = Trainer::parameterSelectionTrain(model, model_data, training_parameter);
		cout << "Accuracy: " << accuracy << endl;
		Trainer::checkOptimalParameters(accuracy, training_parameter);
		training_parameter.C += c_value.stepsize;

		if (training_parameter.C == c_value.upper_limit && training_parameter.gamma == gamma_value.upper_limit)
		{
			cout << "----training with C:" << training_parameter.C << " and gamma:" << training_parameter.gamma << "----" <<endl;
			accuracy = Trainer::parameterSelectionTrain(model, model_data, training_parameter);
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
			accuracy = Trainer::parameterSelectionTrain(model, model_data, training_parameter);
			cout << "Accuracy: " << accuracy << endl;
			Trainer::checkOptimalParameters(accuracy, training_parameter);
			training_parameter.C = c_value.lower_limit;
			training_parameter.gamma += gamma_value.stepsize;
		}

	}

}

double Trainer::parameterSelectionTrain(Model & model, DataFile & train, svm_parameter & custom_parameter)
{
	model.setParameter(custom_parameter);
	//Trainer::train(model, train);
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

	double minimum_c_value = 0.01; //0.01
	double maximum_c_value = 10000; //10000
	double minimum_gamma_value = 0.0001; // 0.001
	double maximum_gamma_value = 10; // 100
	/* Find initial center point for C and gamma */
	double initial_c_value = (minimum_c_value + maximum_c_value) / 2;
	double initial_gamma_value = (minimum_gamma_value + maximum_gamma_value) / 2;
	/* Convert to log scale and cross validate the results */
	cout << "Initial values: " << initial_c_value << " " << initial_gamma_value << endl;
	double initial_c_log_value = calculateLog(maximum_c_value, minimum_c_value, initial_c_value);
	double initial_gamma_log_value = calculateLog(maximum_gamma_value, minimum_gamma_value, initial_gamma_value);
	cout << "Logaritmic values: " << initial_c_log_value << " " << initial_gamma_log_value << endl;
	training_parameter.C = initial_c_log_value;
	training_parameter.gamma = initial_gamma_log_value;
	model.setParameter(training_parameter);
	bool optimal_accuracy_changed = false;
	double accuracy = parameterGridOptimisationTrainer(model, initial_c_log_value, initial_gamma_log_value, optimal_accuracy_changed);
	optimal_accuracy = accuracy;
	cout << "Beginning accuracy: " << accuracy << endl;

	/* Initialize all square C and gamma points */
	double first_square_c_value, second_square_c_value, third_square_c_value, fourth_square_c_value, new_c_value_maximum;
	first_square_c_value = second_square_c_value = third_square_c_value = fourth_square_c_value = initial_c_log_value;
	double first_square_gamma_value, second_square_gamma_value, third_square_gamma_value, fourth_square_gamma_value, new_gamma_value_maximum;
	first_square_gamma_value = second_square_gamma_value = third_square_gamma_value = fourth_square_gamma_value = initial_gamma_log_value;

	vector<double> squares_accuracy_vector;

	int iterations_same_accuracy = 0;
	optimal_accuracy_changed = false;
	while (iterations_same_accuracy < 3)
	{
		cout << "------ New iteration ------" << endl;
		//cout << "Number of iterations with same accuracy: " << iterations_same_accuracy << endl;
		/* Save the current center point of the entire area (square) */
		new_c_value_maximum = first_square_c_value;
		new_gamma_value_maximum = first_square_gamma_value;
		//cout << "Current center point of big square is: " << new_c_value_maximum << " " << new_gamma_value_maximum << endl;
		/* First square calculation */
		first_square_c_value = (first_square_c_value + minimum_c_value) / 2;
		first_square_gamma_value = (first_square_gamma_value + maximum_gamma_value) / 2;
		cout << "Values of C and gamma for 1: " << first_square_c_value << " " << first_square_gamma_value << endl;
		first_square_c_value = calculateLog(maximum_c_value, minimum_c_value, first_square_c_value);
		first_square_gamma_value = calculateLog(maximum_gamma_value, minimum_gamma_value, first_square_gamma_value);
		cout << "Log values of C and gamma for 1: " << first_square_c_value << " " << first_square_gamma_value << endl;
		accuracy = parameterGridOptimisationTrainer(model, first_square_c_value, first_square_gamma_value, optimal_accuracy_changed);
		cout << "Accuracy: " << accuracy << endl;
		squares_accuracy_vector.push_back(accuracy);
		/* Second square calculation */
		second_square_c_value = (second_square_c_value + maximum_c_value) / 2;
		second_square_gamma_value = (second_square_gamma_value + maximum_gamma_value) / 2;
		cout << "Values of C and gamma for 2: " << second_square_c_value << " " << second_square_gamma_value << endl;
		second_square_c_value = calculateLog(maximum_c_value, minimum_c_value, second_square_c_value);
		second_square_gamma_value = calculateLog(maximum_gamma_value, minimum_gamma_value, second_square_gamma_value);
		cout << "Log values of C and gamma for 2: " << second_square_c_value << " " << second_square_gamma_value << endl;
		accuracy = parameterGridOptimisationTrainer(model, second_square_c_value, second_square_gamma_value, optimal_accuracy_changed);
		cout << "Accuracy: " << accuracy << endl;
		squares_accuracy_vector.push_back(accuracy);
		/* Third square calculation */
		third_square_c_value = (third_square_c_value + maximum_c_value) / 2;
		third_square_gamma_value = (third_square_gamma_value + minimum_gamma_value) / 2;
		cout << "Values of C and gamma for 3: " << third_square_c_value << " " << third_square_gamma_value << endl;
		third_square_c_value = calculateLog(maximum_c_value, minimum_c_value, third_square_c_value);
		third_square_gamma_value = calculateLog(maximum_gamma_value, minimum_gamma_value, third_square_gamma_value);
		cout << "Log values of C and gamma for 3: " << third_square_c_value << " " << third_square_gamma_value << endl;
		accuracy = parameterGridOptimisationTrainer(model, third_square_c_value, third_square_gamma_value, optimal_accuracy_changed);
		cout << "Accuracy: " << accuracy << endl;
		squares_accuracy_vector.push_back(accuracy);
		/* Fourth square calculation */
		fourth_square_c_value = (fourth_square_c_value + minimum_c_value) / 2;
		fourth_square_gamma_value = (fourth_square_gamma_value + minimum_gamma_value) / 2;
		cout << "Values of C and gamma for 4: " << fourth_square_c_value << " " << fourth_square_gamma_value << endl;
		fourth_square_c_value = calculateLog(maximum_c_value, minimum_c_value, fourth_square_c_value);
		fourth_square_gamma_value = calculateLog(maximum_gamma_value, minimum_gamma_value, fourth_square_gamma_value);
		cout << "Log values of C and gamma for 4: " << fourth_square_c_value << " " << fourth_square_gamma_value << endl;
		accuracy = parameterGridOptimisationTrainer(model, fourth_square_c_value, fourth_square_gamma_value, optimal_accuracy_changed);
		cout << "Accuracy: " << accuracy << endl;
		squares_accuracy_vector.push_back(accuracy);

		/* Calculate best accuracy */
		int best_accuracy_square = parameterGridOptimisationCheckAccuracy(squares_accuracy_vector);
		int show_accuracy_square = best_accuracy_square + 1;
		cout << "Square with the best accuracy: " << show_accuracy_square << endl;
		if (best_accuracy_square == 0)
		{
			parameterGridOptimisationAssignValues(second_square_c_value, third_square_c_value, fourth_square_c_value, first_square_c_value);
			parameterGridOptimisationAssignValues(second_square_gamma_value, third_square_gamma_value, fourth_square_gamma_value, first_square_gamma_value);
			maximum_c_value = new_c_value_maximum;
			minimum_gamma_value = new_gamma_value_maximum;
		}
		else if (best_accuracy_square == 1)
		{
			parameterGridOptimisationAssignValues(first_square_c_value, third_square_c_value, fourth_square_c_value, second_square_c_value);
			parameterGridOptimisationAssignValues(first_square_gamma_value, third_square_gamma_value, fourth_square_gamma_value, second_square_gamma_value);
			minimum_c_value = new_c_value_maximum;
			minimum_gamma_value = new_gamma_value_maximum;
		}
		else if (best_accuracy_square == 2)
		{
			parameterGridOptimisationAssignValues(first_square_c_value, second_square_c_value, fourth_square_c_value, third_square_c_value);
			parameterGridOptimisationAssignValues(first_square_gamma_value, second_square_gamma_value, fourth_square_gamma_value, third_square_gamma_value);
			minimum_c_value = new_c_value_maximum;
			maximum_gamma_value = new_gamma_value_maximum;
		}
		else
		{
			parameterGridOptimisationAssignValues(first_square_c_value, second_square_c_value, third_square_c_value, fourth_square_c_value);
			parameterGridOptimisationAssignValues(first_square_gamma_value, second_square_gamma_value, third_square_gamma_value, fourth_square_gamma_value);
			maximum_c_value = new_c_value_maximum;
			maximum_gamma_value = new_gamma_value_maximum;
		
		}
		squares_accuracy_vector.clear();
		/* Check if there has been a change in optimal accuracy */
		if (optimal_accuracy_changed)
		{
			iterations_same_accuracy = 0;
			//cout << "Optimal accuracy has changed!" << endl;
			optimal_accuracy_changed = false;
		} 
		else
		{
			++iterations_same_accuracy;
		}
		cout << "------ End iteration ------" << endl;
	}

	cout << "Optimal accuracy with parameters: " << optimal_accuracy << "% C: " << optimal_C_value << " gamma: " << optimal_gamma_value << endl;


}

double Trainer::parameterGridOptimisationTrainer(Model & model, double c_value, double gamma_value, bool & optimal_accuracy_changed)
{
	svm_parameter training_parameter = model.getParameter();
	training_parameter.C = c_value;
	training_parameter.gamma = gamma_value;
	model.setParameter(training_parameter);
	double accuracy = Trainer::crossValidate(model, 150);
	if (accuracy > optimal_accuracy)
	{
		optimal_accuracy = accuracy;
		optimal_C_value = c_value;
		optimal_gamma_value = gamma_value;
		cout << "New optimal accuracy found!" << endl;
		optimal_accuracy_changed = true;
	}
	return accuracy;
}

int Trainer::parameterGridOptimisationCheckAccuracy(vector<double> accuracy_values)
{
	int square = 0;
	auto result = minmax_element(accuracy_values.begin(), accuracy_values.end());
	int minimum = result.first - accuracy_values.begin();
	int maximum = result.second - accuracy_values.begin();
	//cout << "Minimum and maximum: " << minimum << " " << maximum << endl;
	if (accuracy_values[minimum] == accuracy_values[maximum]) return 3;
	return maximum;
}

void Trainer::parameterGridOptimisationAssignValues(double & first, double & second, double & third, double & target_value)
{
	first = target_value;
	second = target_value;
	third = target_value;
}

double Trainer::calculateLog(double maximum, double minimum, double log_variable)
{
	double b = log(maximum / minimum) / (maximum - minimum);
	double a = (maximum / exp(b * maximum));
	double result = (a * exp(b * log_variable));
	return result;
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

void Trainer::featureSelection(DataFile & data_file)
{
	scaleData(data_file);
	int data_length = data_file.getProblemLength();
	int column = 0;
	int amount_of_columns = 0;
	svm_node ** data_node = data_file.getNode();
	mat data_matrix;
	cout << "Length = " << data_length << endl;
	/* Conversion of nodes to matrix */
	for (int i = 0; i < data_length; i++)
	{
		column = 0;
		data_matrix.insert_rows(i, 1);
		while (data_node[i][column].index != -1)
		{
			//cout << data_node[i][column].index << endl;
			if (amount_of_columns <= column)
			{
				data_matrix.insert_cols(column, 1);
				++amount_of_columns;
			}
			++column;
		}
	}
	
	for (int j = 0; j < data_matrix.n_rows; j++)
	{
		for (int k = 0; k < data_matrix.n_cols; k++)
		{
			if (data_node[j][k].index != -1) data_matrix(j, k) = data_node[j][k].value;
			else data_matrix(j, k) = 0;
		}
	}
	
	//data_matrix.print();
	mat selectedFeatures;
	mat projection;
	vec eigenvals;
	data_matrix.print();
	cout << endl;
	//forwardFeatureSelection(&data_matrix, &selectedFeatures, 70);
	forwardFeatureSelection(&data_matrix, &selectedFeatures, 60, false, 0.05);
	data_matrix.print();
	cout << endl;
	selectedFeatures.print();
	/* Conversion of matrix back to nodes */
	
	/*
	svm_node ** new_data_node;
	for (int l = 0; l < selectedFeatures.n_rows; l++)
	{
		new_data_node[l] = new svm_node[selectedFeatures.n_cols];
		for (int m = 0; m < selectedFeatures.n_cols; m++)
		{
			new_data_node[l][m].index = data_node[l][m].index;
			new_data_node[l][m].value = data_matrix(l, m);
		}
	}
	*/
	//getchar();
}

void Trainer::forwardFeatureSelection(mat * data, mat * selectedFeatures, double targetAccuracy, bool usePCA, double varianceThreshold)
{
	double crossValidationAccuracy = 0;

	if (usePCA) {
		mat projection;
		vec eigenvals;

		pca(data, &projection, &eigenvals);

		unsigned int i = 0;
		while (crossValidationAccuracy < targetAccuracy) {
			if (i >= projection.n_cols) {
				cout << "All features are added. Unable to reach target accuracy." << endl;
				break;
			}
			int index_max = eigenvals.index_max();
			selectedFeatures->insert_cols(selectedFeatures->n_cols, projection.col(index_max));
			eigenvals.at(index_max) = -1;
			eigenvals.shed_row(index_max);

			//crossValidate(&selectedFeatures, &crossValidationAccuracy);
			crossValidationAccuracy += 10;
			cout << "Cross validation accuracy: " << crossValidationAccuracy << endl;
			++i;
		}
		if (crossValidationAccuracy >= targetAccuracy) cout << "Target accuracy reached." << endl;
	}
	else { // No PCA
		cout << "number of columns: " << data->n_cols << endl;
		// Prune features
		for (unsigned int i = 0; i < data->n_cols; ++i) {
			//cout << "Variance and duplications on column: " << i << endl;
			// Step 1) Remove low variance features
			//cout << "Variance for column: " << i << " = " << calculateVariance(data->col(i)) << endl;
			if (calculateVariance(data->col(i)) < varianceThreshold)
			{
				//cout << "Column before shed: " << i << endl;
				data->shed_col(i);
				//cout << "Removed" << endl;
			}
			// Step 2) Remove duplicate features
			//cout << "Column " << i << endl;
			//removeDuplicateFeatures(data, i, 0.1);
			//cout << "test" << endl;
		}
		//cout << "Out of for loop!" << endl;
		// Start selection
		while (crossValidationAccuracy < targetAccuracy && !data->is_empty()) {
			double maxAccuracy = 0;
			unsigned int maxAccuracyId = 0;

			for (unsigned int i = 0; i < data->n_cols; ++i) {
				selectedFeatures->insert_cols(selectedFeatures->n_cols, data->col(i));
				//crossValidate(&selectedFeatures, &crossValidationAccuracy);
				if (crossValidationAccuracy > maxAccuracy) {
					maxAccuracy = crossValidationAccuracy;
					maxAccuracyId = i;
				}
				else selectedFeatures->shed_col(selectedFeatures->n_cols - 1);
			}
			crossValidationAccuracy += 10;
			data->shed_col(maxAccuracyId);
		}
	}
}

double Trainer::calculateVariance(vec feature)
{
	double variance = 0;
	if (feature.n_elem > 1) {

		double mean = accu(feature) / feature.n_elem;
		vec submean(feature);

		submean.for_each([mean](arma::vec::elem_type& val) { val -= mean; val = pow(val, 2); });
		variance = accu(submean) / (submean.n_elem - 1);
	}
	return variance;
}

void Trainer::removeDuplicateFeatures(mat * data, unsigned int id, double margin)
{
	vec original = data->col(id);
	for (unsigned int i = 0; i < data->n_cols; ++i) {
		if (i == id) continue;
		vec comp = data->col(i);
		for (unsigned int j = 0; j < original.n_elem; ++j) {
			if (original.at(j) < (comp.at(j) - margin) || original.at(j) > (comp.at(j) + margin)) {
				data->shed_col(i);
				break;
			}
		}
	}
}

void Trainer::pca(mat * data, mat * projection, vec * eigenvals)
{
	mat coeff;

	princomp(coeff, *projection, *eigenvals, *data);

	data->print();
	cout << endl;
	coeff.print();
	cout << endl;
	eigenvals->print();
	cout << endl;
	projection->print();
	cout << endl;
}

void Trainer::scaleData(DataFile & data_file)
{
	int row_length = data_file.getProblemLength();
	svm_node ** data_node = data_file.getNode();
	int column = 0;
	double maximum = 0;
	double minimum = data_node[0][0].value;
	for (int i = 0; i < row_length; i++)
	{
		column = 0;
		while (data_node[i][column].index != -1)
		{
			if (minimum > data_node[i][column].value) minimum = data_node[i][column].value;
			if (maximum < data_node[i][column].value) maximum = data_node[i][column].value;
			++column;
		}
	}
	//cout << minimum << " " << maximum << endl;
	double range = maximum - minimum;
	if (range == 0) range = minimum;
	//cout << "Range and minimum: " << range << " " << minimum << endl;
	data_file.setDataMinimum(minimum);
	data_file.setDataRange(range);
	for (int i = 0; i < row_length; i++)
	{
		column = 0;
		while (data_node[i][column].index != -1)
		{
			if (range != minimum) data_node[i][column].value = (data_node[i][column].value - minimum) / range;
			else data_node[i][column].value = data_node[i][column].value / range;
			++column;
		}
	}
}


Trainer::~Trainer()
{
}
