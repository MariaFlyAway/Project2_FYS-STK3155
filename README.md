# Project 2 FYS-STK3155

## Description
This project aims to find the best fit for simple polynomials and the Franke function using a neural net and various gradient descent methods, as well as classifying benign and malignant tumors from the Wisconsin breast cancer dataset using a neural net and logistic regression.

## Project Structure
The project is organized into several directories.

### `Report/`
Contains the report written for this project in pdf-format.

#### `Exploration_of_Machine_Learning_Methods_Applied_to_Regression_and_Classification_Tasks`
The report in pdf format

- **`Figures/`**
  - Contains all figures used in the report.

- **`Additional_plots/`**
  - Contains all additonal figures and plots not included in the report.

### `Code/`
Contains all code used in the project

#### `Gradient_descent/`
Class implementation of all gradient descent methods, as well as implementation of class with figures.
- `gradient_descent`: class implementation of gradient descent methods
- `subtask_a`: notebook with all fits and explorations related to the gradient descent methods

#### `Neural_Net/`
Contains all files related to implementation of the neural net.
- `classification_own_code`: classifies the Wisconsin breast cancer dataset using the neural net class from `neural_net_without_autograd`.
- `classification_scikit_learn`: performs classification of the Wisconsin breast cancer dataset using MLPClassifier from scikit-learn
- `logistic_regression`: fitting of the Wisconsin breast cancer dataset using logistic regression
- `MNIST_dataset.ipynb`: fitting of the MNIST digits dataset.
- `neural_net_without_autograd`: class implementation of neural net performing the backpropagation "by hand"
- `neural_net`: class implementation of a neural net using autograd for backpropagation
- `plotting_results`: plots all results based on data from the `Results`-folder
- `regression_own_code`: fitting of the Franke function using the class implementation from `neural_net_without_autograd`
- `regression_pytorch`: fitting of the Franke function using PyTorch

- **`Results/`**:
  - Contains printouts from gridsearch and scores for different hyperparameter combinations

## License

The data used in this project is licensed under the Creative Commons Attribution 4.0 International Licence

## Contributors
- **Jonas Jørgensen Telle**
- **Marius Torsheim**
- **Maria Klüwer Øvrebø**
