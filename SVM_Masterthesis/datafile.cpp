#include "datafile.h"



DataFile::DataFile()
{
}

DataFile::DataFile(string path)
{
	ifstream myfile(path);
	int success = processFile(myfile);
	if (success != 0)
	{
		//Processing file failed
		cout << "Something went wrong with processing the test file!" << endl;
		cerr << "Error: " << strerror(errno);
	}
	else
	{
		generateProblem();
	}



	
}


DataFile::~DataFile()
{
}

svm_problem & DataFile::getProblem(void)
{
	return svm_problem_datafile;
}

void DataFile::setProblemLength(int custom_problem_length)
{
	problem_length = custom_problem_length;
}

svm_node ** DataFile::getNode(void)
{
	return svm_node_datafile;
}

void DataFile::setNode(svm_node ** custom_node)
{
	svm_node_datafile = custom_node;
}

int DataFile::getProblemLength(void)
{
	return problem_length;
}

vector<double> DataFile::getYlabels(void)
{
	return y_labels;
}

vector<double> DataFile::getIndexNodes(void)
{
	return index_nodes;
}

vector<double> DataFile::getValueNodes(void)
{
	return value_nodes;
}

void DataFile::setYLabels(vector<double> custom_y_labels)
{
	y_labels = custom_y_labels;
}

double DataFile::getDataRange()
{
	return data_range_minimum_maximum;
}

void DataFile::setDataRange(double range)
{
	data_range_minimum_maximum = range;
}

double DataFile::getDataMinimum()
{
	return data_minimum_value;
}

void DataFile::setDataMinimum(double minimum)
{
	data_minimum_value = minimum;
}

int DataFile::processFile(ifstream & myfile)
{
	string line;
	string lineSubstring;
	int position = 0;
	int positionSubstring = 0;
	int length = 0;
	double index = 0;
	double value = 0;
	char spaceChar;
	double y = 0;
	problem_length = 0;

	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			++problem_length;
			position = 0;
			for (int i = 0; i <= line.length(); i++)
			{
				spaceChar = line[i];
				if (isspace(spaceChar))
				{

					//read every part between " " and the next " "
					lineSubstring = line.substr(position, i - position);
					if (position == 0) //get all labels
					{
						//conversion of 1 or -1 to double
						y = atof(lineSubstring.c_str());
						y_labels.push_back(y);
					}
					else //get nodes of a certain label
					{
						positionSubstring = lineSubstring.find(":");
						index = stoi(lineSubstring.substr(0, positionSubstring));
						value = stoi(lineSubstring.substr(positionSubstring + 1, lineSubstring.length()));
						index_nodes.push_back(index);
						value_nodes.push_back(value);
					}

					position = i + 1;
				}

			}

			lineSubstring = line.substr(position, line.length());
			int previous_value_substring = positionSubstring;
			positionSubstring = lineSubstring.find(":");
			if (positionSubstring < (line.length() - position))
			{
				index = stoi(lineSubstring.substr(0, positionSubstring));
				value = stoi(lineSubstring.substr(positionSubstring + 1, lineSubstring.length()));
				index_nodes.push_back(index);
				value_nodes.push_back(value);
			}
			//Indicate end of x node
			index_nodes.push_back(-1);
		}
		myfile.close();
		return 0;
	}
	else
	{
		cout << "Unable to open file" << endl;
		return 1;
	}
}

int DataFile::processFileNoLabels(string path)
{
	ifstream myfile(path);
	string line;
	string lineSubstring;
	int position = 0;
	int positionSubstring = 0;
	int length = 0;
	int index = 0;
	int value = 0;
	char spaceChar;
	double y = 0;
	problem_length = 0;

	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			++problem_length;
			position = 0;
			for (int i = 0; i < line.length(); i++)
			{
				spaceChar = line[i];
				if (isspace(spaceChar))
				{

					//read every part between " " and the next " "
					lineSubstring = line.substr(position, i - position);
					positionSubstring = lineSubstring.find(":");
					index = stoi(lineSubstring.substr(0, positionSubstring));
					value = stoi(lineSubstring.substr(positionSubstring + 1, lineSubstring.length()));
					index_nodes.push_back(index);
					value_nodes.push_back(value);
					position = i + 1;
				}
			}
			//Indicate end of x node
			index_nodes.push_back(-1);
		}
		myfile.close();
		generateProblem();
		return 0;
	}
	else
	{
		cout << "Unable to open file" << endl;
		return 1;
	}
}

int DataFile::generateProblem(void)
{
	svm_problem_datafile.l = problem_length; // number of rows in training file
	svm_problem_datafile.y = new double[problem_length]; //label each row

	/* Generating all labels for each row */
	int labelCount = 0;
	for (const double& i : y_labels)
	{
		svm_problem_datafile.y[labelCount] = i;
		++labelCount;
	}

	/* Generating problem */
	int number_of_columns = ceil(index_nodes.size() / problem_length);

	svm_node_datafile = new svm_node*[problem_length];
	int collumn = 0;
	int row = 0;
	int value_counter = 0;
	svm_node_datafile[row] = new svm_node[number_of_columns];
	for (const int& j : index_nodes)
	{
		svm_node_datafile[row][collumn].index = j;
		//cout << "Index: " << j << endl;
		if (j != -1)
		{
			svm_node_datafile[row][collumn].value = value_nodes[value_counter];
			//cout << "Value: " << value_nodes[value_counter] << endl;
			++collumn;
			++value_counter;
		}
		else
		{
			collumn = 0;
			++row;
			svm_node_datafile[row] = new svm_node[number_of_columns];
		}
	}

	svm_problem_datafile.x = svm_node_datafile;
	return 0;
}

int DataFile::generateProblemFromNodes(void)
{
	svm_problem_datafile.l = problem_length; // number of rows in training file
	
	svm_problem_datafile.y = new double[problem_length]; //label each row
	/* Generating all labels for each row */
	int labelCount = 0;
	for (const double& i : y_labels)
	{
		svm_problem_datafile.y[labelCount] = i;
		++labelCount;
	}
	svm_problem_datafile.x = svm_node_datafile;
	return 0;
}
