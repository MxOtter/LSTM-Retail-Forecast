# LSTM Sales Prediction for Retail Data

## Overview
This project uses a Long Short-Term Memory (LSTM) neural network to predict future sales based on historical sales data. The model is trained on retail sales data and uses time-series forecasting techniques to predict future sales for a single item/store combination.

The goal of the project is to demonstrate how LSTM models can be applied to retail sales prediction using publicly available data.

## Setup Instructions

1. **Install Dependencies**

To run this project, you'll need to install several Python libraries. You can easily install them using 'pip'. The required libraries are listed in the 'requirements.txt' file. To install all dependencies, run the following command:

```bash
pip install -r requirements.txt
```

The required packages are:

- 'pandas': for data manipulation and loading CSV files.
- 'numpy': for numerical operations.
- 'matplotlib': for plotting the results.
- 'scikit-learn': for data scaling.
- 'tensorflow': for building and training the LSTM model.

**Please ensure you are in the directory containing the 'requirements.txt' file to properly install all necessary packages.**

2. **Data Files**

This project requires two CSV data files:

- 'sales_train_validation.csv': contains historical sales data.
- 'calendar.csv': contains calendar data that merges with the sales data.

These files should be placed in the same directory as the Python script ('LSTM_Final.py') for the code to work properly.

3. **Running the Code**

Once you have installed the dependencies and placed the data files in the same directory, you can run the LSTM model by executing the Python script:

```bash
python LSTM_Final.py
```

4. **Output**

The script will train the LSTM model on the sales data and produce the following outputs:

- Plot 1: a graph showing the actual vs. predicted sales values for the item/store combination.
- Plot 2: a graph showing the training and validation loss over the epochs.

These plots will help visualize how well the model is performing and whether the learning process is converging over time.

## Code Explanation

1. **Data Preprocessing**

The data preprocessing steps include: 

- Loading the 'sales_train_validation.csv' and 'calendar.csv' files.
- Converting the sales data from wide format to long format.
- Merging the sales data with the calendar data for time alignment.
- Normalizing the sales data using MinMaxScaler to scale the values between 0 and 1.
- Creating rolling sequences for the LSTM input.

2. **Model Architecture**

The LSTM model consists of:

- Two LSTM layers with 64 and 32 units, respectively, using the 'tanh' activation function.
- Dropout layer with a rate of 0.2 to reduce overfitting.
- Dense layer with a single output unit for predicting sales.

The model is compiled using the Adam optimizer and Mean Squared Error (MSE) as the loss function. We also use Mean Absolute Error (MAE) and a custom Root Mean Squared Error (RMSE) metric for evaluation.

3. **Learning Rate Scheduling**

A custom learning rate scheduler is used to reduce the learning rate by a factor of 0.1 every 10 epochs. This helps the model converge more effectively.

4. **Training the Model**

The model is trained for 50 epochs using a batch size of 32. A validation split of 0.2 is used to evaluate the model's performance on unseen data during training.

5. **Prediction and Visualization**

Once the model is trained, it makes predictions on the sales data, and the following plots are generated:

- Actual vs. Predicted Sales: this graph compares the model's predicted sales values with the actual sales values from the dataset.
- Training & Validation Loss: this graph shows the loss during the training process, helping you monitor the model's learning progress.

## Files Included:

- 'LSTM_Final.py': the main script that implements the LSTM model, trains it, and generates the predictions and plots.
- 'sales_train_validation.csv': samples sales data (can be replaced with dataset for real-world applications).
- 'calendar.csv': calendar data used for merging with the sales data.

## License:

This code is free to use, modify, and adapt for your personal projects. If you adapt the code, please ensure proper attribution.

## requirements.txt:

This 'requirements.txt' file includes all the necessary Python libraries for the project:

```nginx
pandas
numpy
matplotlib
scikit-learn
tensorflow
```

## Conclusion

This project demonstrates how to use LSTM networks for time-series forecasting on retail sales data. With the modular code and clear explanations, you can easily adapt this project for other similar forecasting tasks or apply it to your own datasets.

