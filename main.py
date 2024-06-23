import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from createModel import create_and_train_model
from evaluateModel import evaluate_model
from plotting import plot_training_metrics, plot_given_vs_predicted
from error import calculate_errors_all, calculate_squared_errors

def main():
    # Load data from Excel files
    datax = pd.read_excel('C:/Users/USER/Desktop/2 INFO 01/PFA2/coding/predictiveModel/MASTERs_DATASET_MODFIEDx.xlsx')
    datay = pd.read_excel('C:/Users/USER/Desktop/2 INFO 01/PFA2/coding/predictiveModel/MASTERs_DATASET_MODFIEDy.xlsx')

    # Split data into input variables (X) and target variables (Y)
    X = datax[['g0', 'Esat']]
    Y = datay[['P0', 'T1', 'P1', 'T2', 'P2', 'T3', 'P3', 'T4', 'P4', 'T5', 'P5', 'T6']]

    # Split data into training, validation, and test sets
    X_train, temp_X, Y_train, temp_Y = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(temp_X, temp_Y, test_size=0.5, random_state=42)

    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # Data Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the model
    model, history = create_and_train_model(X_train_scaled, Y_train, X_val_scaled, Y_val, 350)

    # Evaluate the model
    predictions, loss, mae, mse, rmse, r_squared, adj_r_squared = evaluate_model(model, X_test_scaled, Y_test)

    # Create a dictionary with the metrics values
    metrics_values = {
        'Loss': loss,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R-squared': r_squared,
        'Adj R-squared': adj_r_squared
    }



     # Calculate errors for all predictions
    errors = calculate_errors_all(Y_test, predictions)

    # Calculate squared errors for each sample independently
    squared_errors_for_samples = calculate_squared_errors(Y_test, predictions)
    
    # Create a DataFrame to store errors
    errors_df = pd.DataFrame(errors, columns=['MAE', 'RMSE', 'R2'])

    # Create a DataFrame to store the squared errors
    errors_df['Squared errors'] = squared_errors_for_samples


    # Add input data for reference if necessary
    errors_df[['g0', 'Esat']] = X



    # Plotting code...
    plot_training_metrics(history, metrics_values)
    plot_given_vs_predicted(X_test, Y_test, predictions, 4)

if __name__ == "__main__":
    main()