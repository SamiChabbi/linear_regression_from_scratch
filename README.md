# Linear Regression from Scratch

This project implements linear regression from scratch in Python, showcasing fundamental concepts of machine learning. Linear regression is a foundational algorithm used for predicting a continuous target variable based on one or more input features.

## Overview

- **Implementation from Scratch:** This project demonstrates a step-by-step implementation of linear regression without relying on external libraries for the core functionality. It serves as an educational resource for understanding the inner workings of linear regression.

- **Dataset Generation:** The implementation includes the use of synthetic data generated using scikit-learn's `make_regression` function. The dataset is visualized using matplotlib to provide insight into the relationship between input features and the target variable.

- **Model Initialization:** The model includes the addition of a bias term by appending a column of ones to the input matrix. Parameters (`theta`) are initialized randomly.

- **Cost Function and Gradient Descent:** The cost function, based on the mean squared error, is employed to measure the performance of the model. Gradient descent is then utilized to iteratively adjust model parameters, minimizing the cost and improving prediction accuracy.

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/SamiChabbi/linear_regression_from_scratch.git
    ```

2. Install the required dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

3. Navigate to the project directory:

    ```bash
    cd linear_regression_from_scratch
    ```

4. Run the script:

    ```bash
    python linear_regression.py
    ```

## Results

- The initial model is visualized alongside the dataset, demonstrating the need for parameter adjustment.

- After training through gradient descent, the final model is plotted, showcasing improved alignment with the data points.

## Next Steps

- Experiment with different datasets and hyperparameters to deepen understanding.
- Explore extensions to handle multiple features and enhance the algorithm's versatility.
- Consider incorporating regularization techniques to prevent overfitting.

## Important Note

This implementation is designed for educational purposes to provide insights into the foundational concepts of linear regression. While it may not be as optimized as existing libraries, it serves as a valuable learning tool.

Feel free to explore, experiment, and learn from the code. Happy coding!
