import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np

def plot_training_metrics(history, metrics_values):
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 15))

    # Flatten the axes for easier indexing
    axes = axes.flatten()

    # Plot MAE
    axes[0].plot(history.history['mae'], label='Train')
    axes[0].plot(history.history['val_mae'], label='Validation')
    axes[0].set_title("Evolution of Mean Absolute Error (MAE) during Training")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MAE')
    axes[0].legend()

    # Plot MSE
    axes[1].plot(history.history['mse'], label='Train')
    axes[1].plot(history.history['val_mse'], label='Validation')
    axes[1].set_title('Evolution of Mean Squared Error (MSE) during Training')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].legend()

    # Plot Loss
    axes[2].plot(history.history['loss'], label='Train')
    axes[2].plot(history.history['val_loss'], label='Validation')
    axes[2].set_title('Evolution of Loss during Training')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()

    # Plot Accuracy
    axes[3].plot(history.history['accuracy'], label='Train')
    axes[3].plot(history.history['val_accuracy'], label='Validation')
    axes[3].set_title('Evolution of Accuracy during Training')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Accuracy')
    axes[3].legend()

    # Plot RMSE with additional parameters and calculated metrics in a table
    axes[4].plot(range(1, len(history.history['loss']) + 1), history.history['loss'], label='Train')
    axes[4].plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], label='Validation')
    axes[4].set_title('Evolution of Loss during Training')
    axes[4].set_xlabel('Epoch')
    axes[4].set_ylabel('Loss')
    axes[4].legend()

    # Remove the unused subplot
    fig.delaxes(axes[5])

    # Create a new table subplot in the space of the removed subplot
    table_ax = fig.add_subplot(3, 2, 6)

    # Additional parameters and calculated metric values
    metrics_df = pd.DataFrame(metrics_values.items(), columns=['Parameter', 'Value'])

    # Display the metrics in a table
    table = table_ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns,
                            cellLoc='center', loc='center', colWidths=[0.3, 0.3])

    # Adjust font size and style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1])
    
    # Hide the axis of the table subplot
    table_ax.axis('off')

    # Adjust subplot spacing
    fig.tight_layout()

    # Show the plot
    plt.show()


def plot_given_vs_predicted(X_test, Y_test, predictions, num_samples_to_plot):
    # Calculate the number of rows and columns for the subplot grid
    num_rows = (num_samples_to_plot + 1) // 2
    num_columns = 2

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(12, 10))

    # Flatten the axes for easier indexing
    axes = axes.flatten()

    # Loop through the number of samples
    for i in range(num_samples_to_plot):
        # Choose a random index for validation
        validation_index = random.randint(0, len(Y_test) - 1)

        # Data for the specific case
        donnees_du_cas = Y_test.iloc[validation_index]

        # Assign each value to a separate variable
        P0, T1, P1, T2, P2, T3, P3, T4, P4, T5, P5, T6 = donnees_du_cas

        # Extracted points
        points = [
            (0, -T6), (P5, -T5), (P4, -T4), (P3, -T3),
            (P2, -T2), (P1, -T1), (P0, 0), (P1, T1),
            (P2, T2), (P3, T3), (P4, T4), (P5, T5), (0, T6)
        ]

        # Separate x and y coordinates
        x_values, y_values = zip(*points)

        # Predicted values for the specific case
        predicted_values = predictions[validation_index]

        PP0, PT1, PP1, PT2, PP2, PT3, PP3, PT4, PP4, PT5, PP5, PT6 = predicted_values

        ppoints = [
            (0, -PT6), (PP5, -PT5), (PP4, -PT4), (PP3, -PT3),
            (PP2, -PT2), (PP1, -PT1), (PP0, 0), (PP1, PT1),
            (PP2, PT2), (PP3, PT3), (PP4, PT4), (PP5, PT5), (0, PT6)
        ]

        # Separate x and y coordinates
        px_values, py_values = zip(*ppoints)


        # Calculate the error for the specific case
        error = np.sum((donnees_du_cas - predicted_values) ** 2) / np.sum(predicted_values ** 2)

        # Plot the given points and predicted values
        axes[i].plot(y_values, x_values, marker='', linestyle='-', label='Given Points')
        axes[i].plot(py_values, px_values, marker='', linestyle='--', label='Predicted Values')

        # Add labels and title
        axes[i].set_xlabel('Time[ps]')
        axes[i].set_ylabel('Power[mW]')
        axes[i].set_title(f'Sample {i+1} - ERROR: {error:.4f}')

        # Display legend
        axes[i].legend()


         # Add text annotations for the values of g0 and Esat
        g0_value = X_test.iloc[validation_index]['g0']
        Esat_value = X_test.iloc[validation_index]['Esat']
        axes[i].text(0.5, 0.5, f'g0={g0_value}, Esat={Esat_value}', 
             transform=axes[i].transAxes, ha='center', va='center', fontsize=10, color='blue')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()



