import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import neural_network as nn
import joblib

# Data preparation
def prepare_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=data.columns[:4])  # Drop the first four columns
    data.rename(columns={
        'field1': 'Temperature', 
        'field2': 'Humidity',
        'field3': 'X-axis', 
        'field4': 'Y-axis'
    }, inplace=True)
    return data

# Data split and normalization
def split_and_normalize_data(data):
    X = data.drop(columns=['X-axis'])
    y = data['X-axis'].values
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7, random_state=12)
    X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, train_size=0.5, random_state=17)
    return X_train, X_test, X_val, y_train, y_test, y_val

# Plot features
def plot_features(data):
    features = ['Temperature', 'Humidity', 'X-axis', 'Y-axis']
    labels = ['Temperature (°C)', 'Humidity (%)', 'Degrees (°)', 'Degrees (°)']

    for feature, label in zip(features, labels):
        plt.figure()
        plt.plot(data[feature])
        plt.xlabel('Number of Records')
        plt.ylabel(label)
        plt.title(feature)
        plt.grid(True)
        plt.show()

    return

# Calculate Pearson Correlation Coefficient
def calculate_pearson(data):
    correlations = {
        'Temperature': pearsonr(data['X-axis'], data['Temperature']),
        'Humidity': pearsonr(data['X-axis'], data['Humidity']),
        'Y-axis': pearsonr(data['X-axis'], data['Y-axis']),
    }
    for key, (coeff, _) in correlations.items():
        print(f"Pearson Correlation Coefficient for {key}: {coeff:.2f}")
    return

# Train and evaluate MLPRegressor with varying epochs
def evaluate_epochs(X_train, X_val, y_train, y_val, hidden_layers):
    epoch_numbers = range(10, 160, 10)
    mse_scores = []

    for num_epochs in epoch_numbers:
        model = nn.MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=num_epochs, random_state=0,
                                activation='relu', solver='adam', batch_size=1, learning_rate_init=1e-5)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        mse = metrics.mean_squared_error(y_val, predictions)
        mse_scores.append(mse)
        print(f"Epochs: {num_epochs}, MSE: {mse}")

    plt.plot(epoch_numbers, mse_scores, marker='o')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Optimal Number of Epochs')
    plt.grid(True)
    plt.show()

# Train and evaluate MLPRegressor with varying learning rates
def evaluate_learning_rate(X_train, X_val, y_train, y_val, hidden_layers):
    learning_rates = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    mse_scores = []

    for lr in learning_rates:
        model = nn.MLPRegressor(hidden_layer_sizes=hidden_layers, learning_rate_init=lr, max_iter=300, random_state=0,
                                activation='relu', solver='adam', batch_size=1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        mse = metrics.mean_squared_error(y_val, predictions)
        mse_scores.append(mse)

    plt.plot(learning_rates, mse_scores, marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Optimal Learning Rate')
    plt.grid(True)
    plt.show()

# Train final MLP model
def train_final_model(X_train, y_train, X_test, y_test):
    model = nn.MLPRegressor(hidden_layer_sizes=(4, 4, 4), max_iter=300, random_state=121,
                            activation='relu', solver='adam', batch_size=1, learning_rate_init=1e-4)
    model.fit(X_train, y_train)
    joblib.dump(model, 'mlp_model.pkl')

    predictions = model.predict(X_test)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, predictions)
    mae = metrics.mean_absolute_error(y_test, predictions)

    print(f'MSE: {mse}, RMSE: {rmse}, R-squared: {r2}, MAE: {mae}')

    plt.plot(predictions, color='r', label='MLP')
    plt.plot(y_test, color='b', label='Actual')
    plt.xlabel('Number of Data Points')
    plt.ylabel('Inclination (°)')
    plt.legend()
    plt.grid(True)
    plt.show()

    pearson_coeff, _ = pearsonr(y_test, predictions)
    print(f"Pearson Correlation Coefficient: {pearson_coeff:.2f}")

    errors = y_test - predictions
    plt.plot(errors)
    plt.xlabel('Number of Data Points')
    plt.ylabel('Errors (°)')
    plt.grid(True)
    plt.show()

# Main Execution
if __name__ == "__main__":
    data_file = 'filepath.csv'
    data = prepare_data(data_file)
    
    # Set this flag to choose the mode (e.g., plotting, training, etc.)
    mode = 0

    if mode == 0:
        plot_features(data)
        calculate_pearson(data)

    X_train, X_test, X_val, y_train, y_test, y_val = split_and_normalize_data(data)
    
    if mode == 1:
        evaluate_epochs(X_train, X_val, y_train, y_val, hidden_layers=(15,) * 10)
    elif mode == 2:
        evaluate_learning_rate(X_train, X_val, y_train, y_val, hidden_layers=(15,) * 10)
    elif mode == 3:
        train_final_model(X_train, y_train, X_test, y_test)
    # Add additional modes for other functionality (e.g., shape analysis, etc.)
# Compare data for May
if mode == 6:
    compare_data = pd.read_csv('D://New folder/newforcomprison2.csv')
    compare_data = compare_data.drop(columns=compare_data.columns[[0, 1, 2]])
    compare_data.rename(columns={'field1': 'Temperature', 'field2': 'Humidity',
                                 'field3': 'X-axis', 'field4': 'Y-axis'}, inplace=True)

    # Normalize comparison data
    x_compare = compare_data.drop(columns=['X-axis']).copy()
    x_compare = scaler.transform(x_compare)
    y_compare = compare_data['X-axis'].values

    # Load trained model
    pl = nn.MLPRegressor(max_iter=300, random_state=121, hidden_layer_sizes=(5, 5, 5),
                         activation='relu', solver='adam', batch_size=1, learning_rate_init=1e-4)
    pl.fit(xtrain, ytrain)

    y_pred_compare = pl.predict(x_compare)
    test_mse = metrics.mean_squared_error(y_compare, y_pred_compare)
    rmse = np.sqrt(test_mse)
    r2 = r2_score(y_compare, y_pred_compare)
    mae = metrics.mean_absolute_error(y_compare, y_pred_compare)

    print(f'MSE: {test_mse}, RMSE: {rmse}, R-squared: {r2}, MAE: {mae}')

    # Plotting results with time tag
    if timetag == 1:
        compare_data_time = pd.read_csv('D://New folder/newforcomprison3.csv')
        compare_data_time = compare_data_time.drop(columns=compare_data_time.columns[[0, 1, 3, 4, 5, 6]])
        compare_data_time['created_at'] = pd.to_datetime(compare_data_time['created_at'])
        compare_data_time.set_index('created_at', inplace=True)

        plt.figure(figsize=(12, 6))
        plt.plot(compare_data_time.index, y_pred_compare, color='r', label='MLP')
        plt.plot(compare_data_time.index, y_compare, color='b', label='LARA')
        plt.xlabel("Date")
        plt.ylabel("Inclination (°)")
        plt.title("Comparison for May")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Histogram of errors with normal distribution fit
    if stv == 1:
        errors = y_compare - y_pred_compare
        plt.figure(figsize=(8, 6))
        plt.hist(errors, bins=10, density=True, alpha=0.6, color='blue')
        mu, sigma = np.mean(errors), np.std(errors)
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
        plt.plot(x, (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)),
                 color='red', linewidth=2)
        plt.xlabel('Errors')
        plt.ylabel('Probability Density')
        plt.title('Histogram of Errors with Fitted Normal Distribution')
        plt.show()

        print(f"Mean Squared Error (MSE): {test_mse}")
        print(f"Mean of Errors: {mu}")
        print(f"Standard Deviation of Errors: {sigma}")

# Standard deviation and Monte Carlo bell curve of test data
if mode == 7:
    pl = nn.MLPRegressor(max_iter=300, random_state=121, hidden_layer_sizes=(5, 5, 5),
                         activation='relu', solver='adam', batch_size=1, learning_rate_init=1e-4)
    pl.fit(xtrain, ytrain)
    y_pred_test = pl.predict(xtest)
    errors = ytest - y_pred_test

    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=10, density=True, alpha=0.6, color='blue')
    mu, sigma = np.mean(errors), np.std(errors)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
    plt.plot(x, (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)),
             color='red', linewidth=2)
    plt.xlabel('Errors')
    plt.ylabel('Probability Density')
    plt.title('Histogram of Errors with Fitted Normal Distribution')
    plt.show()

    print(f"Mean Squared Error (MSE): {test_mse}")
    print(f"Mean of Errors: {mu}")
    print(f"Standard Deviation of Errors: {sigma}")

# Daily MSE and standard deviation comparison
if mode == 8:
    pl = nn.MLPRegressor(max_iter=300, random_state=121, hidden_layer_sizes=(5, 5, 5),
                         activation='relu', solver='adam', batch_size=1, learning_rate_init=1e-4)
    pl.fit(xtrain, ytrain)

    data_m = pd.read_csv('D://New folder/thelast.csv')
    data_m.rename(columns={'field1': 'Temperature', 'field2': 'Humidity',
                           'field3': 'X-axis', 'field4': 'Y-axis'}, inplace=True)
    data_m['time'] = pd.to_datetime(data_m['time'])
    grouped_data = data_m.groupby(data_m['time'].dt.date)

    days = []
    mses = []
    sigmas = []

    for day, day_data in grouped_data:
        X = day_data.drop(['time', 'X-axis'], axis=1)
        X = scaler.transform(X)
        y = day_data['X-axis']

        y_pred = pl.predict(X)
        mse = metrics.mean_squared_error(y, y_pred)
        errors = y_pred - y

        days.append(day)
        mses.append(mse)
        sigmas.append(np.std(errors))

    # Plot MSE values
    plt.figure(figsize=(10, 6))
    plt.plot(days, mses, marker='o', label='MLP')
    plt.axhline(y=0.00058, color='r', linestyle='--', label='LARA Threshold')
    plt.xlabel('Day')
    plt.ylabel('Mean Squared Error')
    plt.title('Mean Squared Error by Day')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot standard deviation values
    plt.figure(figsize=(10, 6))
    plt.plot(days, sigmas, marker='o', label='MLP')
    plt.axhline(y=0.00058, color='r', linestyle='--', label='LARA Threshold')
    plt.xlabel('Day')
    plt.ylabel('Standard Deviation')
    plt.title('Standard Deviation by Day')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()
