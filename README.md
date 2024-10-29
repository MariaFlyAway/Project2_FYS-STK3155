# Project 2 FYS-STK3155

## Description
This project aims to find the best fit for the Franke function using a neural net and various gradient descent methods, as well as classifying benign and malignant tumors from the Wisconsin breast cancer dataset using a neural net and logistic regression.

## Project Structure
The project is organized into several directories.

### `Report/`
Contains the report written for this project in pdf-format.

#### `Report name`
The report

- **`Figures/`**
  - Contains all figures used in the report.

- **`Additional figures/`**
  - Contains all additonal figures and plots not included in the report.

### `Code/`
Contains all code used in the project

#### `Gradient_descent/`
Class implementation of all gradient descent methods, as well as implementation of class with figures.
- `gradient_descent`: class implementation of gradient descent methods

#### `Neural_Net/`
Contains all files related to implementation of the neural net.
- `MNIST_dataset.ipynb`: fitting of the MNIST digits dataset.
- `classification_own_code.ipynb`: classifies the Wisconsin breast cancer dataset using the neural net class from `neural_net_without_autograd`.
- `neural_net.py`: class implementation of a neural net using autograd for backpropagation
- `neural_net_without_autograd`: class implementation of neural net performing the backpropagation "by hand"
- `plotting_results.ipynb`: plots all results based on data from the `Results`-folder
- `regression_own_code.ipynb`: fitting of the Franke function using the class implementation from `neural_net_without_autograd`
- `regression_pytorch`: fitting of the Franke function using PyTorch
- `classification_scikit_learn`: performs classification of the Wisconsin breast cancer dataset using MLPClassifier from scikit-learn

- **`Results/`**:
  - Contains printouts from gridsearch and scores for different hyperparameter combinations

## License

The data used in this project is licensed under the Creative Commons Attribution 4.0 International Licence

## Contributors
- **Jonas Jørgensen Telle**
- **Marius Torsheim**
- **Maria Klüwer Øvrebø**
