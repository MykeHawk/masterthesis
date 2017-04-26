#include "trainer.h"



Trainer::Trainer()
{
	// Init parameter selection values
	c_value.lower_limit = 0.5;
	c_value.upper_limit = 2;
	c_value.stepsize = 0.5;

	gamma_value.lower_limit = 0.2;
	gamma_value.upper_limit = 1;
	gamma_value.stepsize = 0.2;

}

void Trainer::train(Model & model, DataFile & train, bool parameter_selection, bool feature_selection)
{
	svm_problem training_problem = train.getProblem();
	svm_parameter training_parameter = model.getParameter();
	svm_model * train_model = svm_train(&training_problem, &training_parameter);
	model.setDataFileTrainer(train);
	model.setSvmModel(train_model);

}

void Trainer::crossValidate(Model & model, int cross_validation_parameter)
{
	// Test parameter, needs to be given as parameter to function
	int k_fold_parameter = cross_validation_parameter;

	/* Get data from the model  */
	DataFile datafile_training = model.getDatafile();
	svm_node ** node_datafile_training = datafile_training.getNode();
	svm_model * model_training = model.getSvmModel();
	vector<double> datafile_traininglabels = datafile_training.getYlabels();

	/* Setting up cross validation parameters */
	int crossvalidate_maximum = datafile_training.getProblemLength();
	int crossvalidate_parameter = (crossvalidate_maximum / k_fold_parameter) ;
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

	/* Start of cross validation depending on the number of folds */
	while (bin < k_fold_parameter)
	{
		int testing_row = 0;
		int training_row = 0;
		crossvalidate_testing_bin_nodes[bin] = new svm_node*[crossvalidate_parameter];
		crossvalidate_training_bin_nodes[bin] = new svm_node*[crossvalidate_maximum - crossvalidate_parameter];
		cout << "-- bin -- = " << bin << endl;
		algorithm_iterator_max = (bin + 1) * (crossvalidate_parameter);
		for(int i = 0; i < crossvalidate_maximum; i++)
		{
			if (i >= algorithm_iterator_min && i < algorithm_iterator_max)
			{
				crossvalidate_testing_bin_nodes[bin][testing_row] = node_datafile_training[i];
				++testing_row;
			}
			else
			{
				crossvalidate_training_bin_nodes[bin][training_row] = node_datafile_training[i];
				++training_row;
			}
		} 

		++bin;
		algorithm_iterator_min = bin * (crossvalidate_parameter);
	}

	//Testing if data is correct
	int test = crossvalidate_maximum - crossvalidate_parameter;
	for (int j = 0; j < bin; j++)
	{
		cout << "----------testing bin nodes on bin: " << j << endl;
		for (int k = 0; k < crossvalidate_parameter; k++)
		{
			//cout << "--testing the test bin nodes: " << crossvalidate_testing_bin_nodes[j][k][0].index << endl;
		}
		cout << endl;
		for (int l = 0; l < test; l++)
		{
			cout << "testing the training bin nodes: " << crossvalidate_training_bin_nodes[j][l][0].index << endl;
		}
	}



}

void Trainer::parameterSelection(Model & model, DataFile & train, DataFile & testing_file)
{
	svm_problem train_problem = train.getProblem();
	svm_parameter training_parameter = model.getParameter();

	/* boundaryValue c_value;
	c_value.lower_limit = 0.5;
	c_value.upper_limit = 2;
	c_value.stepsize = 0.5;

	boundaryValue gamma_value;
	gamma_value.lower_limit = 0.2;
	gamma_value.upper_limit = 1;
	gamma_value.stepsize = 0.2; */

	training_parameter.C = c_value.lower_limit;
	training_parameter.gamma = gamma_value.lower_limit;
	double accuracy = 0;
	while (1)
	{
		if (accuracy >= 99)
		{
			cout << "Accuracy is near perfect, stopping the parameter selection" << endl;
			break;
		}
		cout << "----training with C:" << training_parameter.C << " and gamma:" << training_parameter.gamma << "-----" << endl;
		accuracy = Trainer::parameterSelectionTrain(model, train, testing_file, training_parameter);
		cout << "Accuracy: " << accuracy << endl;
		training_parameter.C += c_value.stepsize;

		if (training_parameter.C == c_value.upper_limit && training_parameter.gamma == gamma_value.upper_limit)
		{
			cout << "----training with C:" << training_parameter.C << " and gamma:" << training_parameter.gamma << "----" <<endl;
			Trainer::parameterSelectionTrain(model, train, testing_file, training_parameter);
			cout << "End!" << endl;
			break;
		}

		if (training_parameter.C == c_value.upper_limit && accuracy <= 99)
		{
			cout << "----training with C:" << training_parameter.C << " and gamma:" << training_parameter.gamma << "----" << endl;
			accuracy = Trainer::parameterSelectionTrain(model, train, testing_file, training_parameter);
			training_parameter.C = c_value.lower_limit;
			training_parameter.gamma += gamma_value.stepsize;
		}

	}

}

double Trainer::parameterSelectionTrain(Model & model, DataFile & train, DataFile & testing_file, svm_parameter & custom_parameter)
{
	model.setParameter(custom_parameter);
	Trainer::train(model, train);
	double accuracy = model.predict(testing_file);
	return accuracy;
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


Trainer::~Trainer()
{
}
