import matplotlib.pyplot as plt
import pandas as pd

# Load the ROC data from the CSV file
rocDF = pd.read_csv("rocOutput.csv")

# Create a figure and axis for the plot
plt.figure(figsize=(8, 6))
plt.title("ROC Curves for Different Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

# Plot the ROC curve for each model
for model in rocDF['model'].unique():
    model_data = rocDF[rocDF['model'] == model]
    plt.plot(model_data['fpr'], model_data['tpr'], label=model)

# Add a legend, grid, and diagonal line
plt.legend()
plt.grid(True)
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal dashed line

# Show the plot
plt.show()