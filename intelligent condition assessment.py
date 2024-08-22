import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics, neural_network as nn
from scipy.stats import pearsonr
import joblib

# Function to preprocess data
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(data.columns[[0, 1, 2, 3]], axis=1)
    data.rename(columns={
        'field1': 'Temperature',
        'field2': 'Humidity',
        'field3': 'X-axis',
        'field4': 'Y-axis'
    }, inplace=True)
    return data

# Function to split data into training, validation, and test sets
def split_data(data):
    xsplit = data.drop(columns=['X-axis'])
    ysplit = data['X-axis'].values

    xtrain, xrem, ytrain, yrem = train_test_split(
        xsplit, ysplit, train_size=0.7, random_state=12)
    xtest, xval, ytest, yval = train_test_split(
        xrem, yrem, train_size=0.5, random_state=17)

    return xtrain, xtest, xval, ytrain, ytest, yval

# Function to plot data
def plot_data(data, columns, y_labels):
    for col, y_label in zip(columns, y_labels):
        plt.figure()
        plt.plot(data[col])
        plt.xlabel('Number of Records')
        plt.ylabel(y_label)
        plt.title(col)
        plt.grid(True)
        plt.show()

# Function to calculate and print Pearson correlation
def calculate_pearson(data):
    correlations = {
        "Temperature": pearsonr(data['X-axis'], data['Temperature']),
        "Humidity": pearsonr(data['X-axis'], data['Humidity']),
        "Y-axis": pearsonr(data['X-axis'], data['Y-axis'])
    }
    for key, value in correlations.items():
        print(f"Pearson Correlation Coefficient for {key}: {value[0]:.2f}")

# Function to train and evaluate the MLP model based on number of epochs
def evaluate_epochs(xtrain, ytrain, xval, yval, epoch_numbers, hidden_layer):
    mse_scores = []
    for num_epochs in epoch_numbers:
        model = nn.MLPRegressor(hidden_layer_sizes=hidden_layer, max_iter=num_epochs, random_state=0,
                                activation='relu', solver='adam', batch_size=1, learning_rate_init=1e-5)
        model.fit(xtrain, ytrain)
        predictval = model.predict(xval)
        mse_scores.append(metrics.mean_squared_error(yval, predictval))
        print(f"Epochs: {num_epochs}, MSE: {mse_scores[-1]:.4f}")

    plt.plot(epoch_numbers, mse_scores, marker='o')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Optimal Number of Epochs')
    plt.grid(True)
    plt.show()

# Function to evaluate the model based on learning rates
def evaluate_learning_rates(xtrain, ytrain, xval, yval, learning_rates, hidden_layer):
    mse_scores = []
    for learning_rate in learning_rates:
        model = nn.MLPRegressor(hidden_layer_sizes=hidden_layer, learning_rate_init=learning_rate, max_iter=300,
                                random_state=0, activation='relu', solver='adam', batch_size=1)
        model.fit(xtrain, ytrain)
        predictval = model.predict(xval)
        mse_scores.append(metrics.mean_squared_error(yval, predictval))

    plt.plot(learning_rates, mse_scores, marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Optimal Learning Rate')
    plt.grid(True)
    plt.show()

# Function to train final MLP model and save it
def train_final_model(xtrain, xtest, ytrain, ytest, hidden_layer):
    model = nn.MLPRegressor(max_iter=300, random_state=121, hidden_layer_sizes=hidden_layer, 
                            activation='relu', solver='adam', batch_size=1, learning_rate_init=1e-4)
    model.fit(xtrain, ytrain)
    joblib.dump(model, 'mlp_model.pkl')
    predicttest = model.predict(xtest)
    testmse = metrics.mean_squared_error(ytest, predicttest)
    rmse = np.sqrt(testmse)
    r2 = metrics.r2_score(ytest, predicttest)
    mae = metrics.mean_absolute_error(ytest, predicttest)

    print(f'testmse={testmse:.4f}, RMSE={rmse:.4f}, R-squared={r2:.4f}, MAE={mae:.4f}')

    plt.plot(predicttest, color='r', label='MLP')
    plt.plot(ytest, color='b', label='Actual')
    plt.xlabel('Number of Data')
    plt.ylabel('Inclination (°)')
    plt.legend()
    plt.show()

    return model

# Function to analyze errors
def analyze_errors(ytest, predicttest):
    errors = ytest - predicttest
    plt.hist(errors, bins=10, density=True, alpha=0.6, color='blue')

    mu, sigma = np.mean(errors), np.std(errors)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    plt.plot(x, pdf, color='red', linewidth=2)

    plt.xlabel('Errors')
    plt.ylabel('Probability Density')
    plt.title('Histogram of Errors with Fitted Normal Distribution')
    plt.show()

    print(f"Mean of Errors: {mu:.4f}, Standard Deviation of Errors: {sigma:.4f}")

# Main function
def main():
    file_path = 'file path.csv'
    data = preprocess_data(file_path)
    
    xtrain, xtest, xval, ytrain, ytest, yval = split_data(data)

    # Choose what you want to do
    print("Choose an option:")
    print("0: Plot data and calculate Pearson correlation")
    print("1: Evaluate optimal number of epochs")
    print("2: Evaluate optimal learning rate")
    print("3: Train final model and analyze errors")
    
    choice = int(input("Enter your choice (0-3): "))

    if choice == 0:
        columns_to_plot = ['Temperature', 'Humidity', 'X-axis', 'Y-axis']
        y_labels = ['Temperature (C°)', "RH %", 'Degrees (°)', 'Degrees (°)']
        plot_data(data, columns_to_plot, y_labels)
        calculate_pearson(data)
    
    elif choice == 1:
        epoch_numbers = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        hidden_layer = tuple([15] * 10)
        evaluate_epochs(xtrain, ytrain, xval, yval, epoch_numbers, hidden_layer)
    
    elif choice == 2:
        learning_rates = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        hidden_layer = tuple([15] * 10)
        evaluate_learning_rates(xtrain, ytrain, xval, yval, learning_rates, hidden_layer)
    
    elif choice == 3:
        hidden_layer = (5, 5, 5)
        model = train_final_model(xtrain, xtest, ytrain, ytest, hidden_layer)
        analyze_errors(ytest, model.predict(xtest))
    
    else:
        print("Invalid choice. Please run the program again and select a valid option.")

if __name__ == '__main__':
    main()
