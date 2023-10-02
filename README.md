# Boston Housing Price Prediction

This Python script performs Boston housing price prediction using a traditional linear regression model and a neural network. It utilizes popular machine learning libraries such as TensorFlow, Keras, and scikit-learn.

## Getting Started

### Prerequisites

Make sure you have the following prerequisites installed:

- Python 3.x
- TensorFlow
- Keras
- scikit-learn
- pandas
- matplotlib

You can install these libraries using `pip`:

```shell
pip install tensorflow keras scikit-learn pandas matplotlib
```
## Usage

    Clone this repository:

    shell
```
git clone https://github.com/maurticeoboya/house_price_prediction.git
cd house_price_prediction
```
Run the Python script:

shell
```
    python boston.py
```

Code Details

The code consists of two main parts:

    Traditional Linear Regression:
        Loads the Boston Housing dataset from a CSV file.
        Performs data preprocessing, including scaling using StandardScaler.
        Splits the data into training and testing sets.
        Fits a linear regression model to the training data and evaluates its performance using R-squared.

    Neural Network with Keras:
        Defines a neural network model using Keras with multiple layers.
        Compiles the model using the Adam optimizer and mean squared error loss.
        Trains the model on the training data and evaluates its performance using R-squared.

## Results
    The script prints the R-squared score for both the linear regression model and the neural network, allowing you to compare their performance in predicting    Boston housing prices.

## License
    This project is licensed under the MIT License - see the LICENSE file for details.
## Acknowledgments

    Thanks to the contributors of the TensorFlow, Keras, and scikit-learn libraries for their excellent work.


