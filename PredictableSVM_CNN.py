#%%
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pyts.image import GramianAngularField
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import shutil
# Additional imports for early stopping
import copy

#%%

# Load the dataset
df = pd.read_csv("dataset/EURUSDH4.csv", delimiter="\t")

# Extract the closing prices
closing = df["<CLOSE>"]

# Parameters
window_size = 20  # Example window size
test_size = 0.2   # Test set size


scaler = StandardScaler()

# Create sliding window features and labels
X, y = [], []
for i in range(len(closing) - window_size):
    window = closing[i:i + window_size].values
    target = 1 if closing[i + window_size] > closing[i + window_size - 3] else 0 # Yesterday to Tomorrow
    # Standardize the data
    features = scaler.fit_transform(window[:-1].reshape(-1, 1))
    X.append(features)
    y.append(target)


X = np.array(X).squeeze()
y = np.array(y)

# Step 1: Initial Split into Train_Valid and Test Sets
train_valid_size = 0.8  # 80% for training and validation, 20% for testing
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=(1 - train_valid_size), shuffle=False)

# Step 2: Split Train_Valid into Train and Validation Sets
# train_size = 0.75  # 75% of train_valid for training, 25% for validation
# X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y[:len(X_train_valid)], test_size=(1 - train_size), shuffle=True, random_state=42)



# Train the SVM model first time on train set only
# svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
# svm_model = SVC()
# svm_model.fit(X_train_valid, y_train_valid)
# print("SVM train valid score: ",  svm_model.score(X_train_valid, y_train_valid) )

# Test the SVM model
# svm_predictions = svm_model.predict(X_train_valid)
#
# # Now, we generate GAF images for the test set using pyts
feature_window_size = window_size - 1
gaf = GramianAngularField(image_size=feature_window_size, method='summation')
X_test_gaf = gaf.fit_transform(X_train_valid)


# Directory to be removed and recreated
dir_path = 'gaf_images'

# Check if the directory exists
if os.path.exists(dir_path):
    # Remove the directory and all its contents
    shutil.rmtree(dir_path)

# Now create the directory
os.makedirs(dir_path, exist_ok=True)

for idx, (image, label) in enumerate(zip(X_test_gaf, y_train_valid)):
    class_label = "up" if y_train_valid[idx]==0 else "down"
    np.save(f'gaf_images/{class_label}_{idx}.npy', image)

# GAF images are now saved, and the next step will be CNN training.

#%%
class GAFClassifierCNN(nn.Module):
    def __init__(self):
        super(GAFClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size after the pooling layers
        self.fc1_input_size = 32 * (feature_window_size // 2 // 2) * (feature_window_size // 2 // 2)

        self.fc1 = nn.Linear(self.fc1_input_size, 128)  # Adjust according to the input size
        self.fc2 = nn.Linear(128, 2)  # 2 output classes: predictable, unpredictable

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc1_input_size)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#%%
class GAFDataset(Dataset):
    def __init__(self, image_dir, flip_prob=0.0):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.npy')]
        self.labels = [0 if 'up' in f else 1 for f in self.image_files]
        self.images = [np.load(os.path.join(self.image_dir, f)) for f in self.image_files]
        self.flip_prob = flip_prob  # Probability to flip the label

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.expand_dims(self.images[idx], axis=0)  # Add channel dimension
        label = self.labels[idx]

        # Flip the label with probability flip_prob
        if np.random.rand() < self.flip_prob:
            label = 1 - label  # Flip 0 to 1 or 1 to 0

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# Prepare the dataset and dataloader
gaf_dataset = GAFDataset(image_dir='gaf_images', flip_prob=0.14)# 10% chance to flip the label
train_loader = DataLoader(gaf_dataset, batch_size=32, shuffle=True)

#%%
def trainCNN():
    # Initialize the CNN, loss function, and optimizer
    cnn_model = GAFClassifierCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

    # Training loop with early stopping
    num_epochs = 200
    patience = 10  # Number of epochs with no improvement after which training will be stopped
    best_accuracy = 0.0
    best_model_wts = copy.deepcopy(cnn_model.state_dict())
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        cnn_model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = cnn_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate the number of correct predictions
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        accuracy = correct_predictions / total_predictions * 100

        # Early stopping logic
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_wts = copy.deepcopy(cnn_model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

    # Load the best model weights
    cnn_model.load_state_dict(best_model_wts)

    print("CNN training completed.")
    print(f'Best Accuracy: {best_accuracy:.2f}%')
    return cnn_model, best_accuracy
#%%
def predict_cnn(cnn_model, feature_window):
    # No need to scale the feature_window again since it was done during data preparation

    # Generate GAF image from the feature window
    gaf = GramianAngularField(image_size=feature_window_size, method='summation')
    gaf_image = gaf.fit_transform(feature_window.reshape(1, -1))

    # Prepare the GAF image for the CNN
    gaf_image = torch.tensor(gaf_image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension only

    # Use CNN to predict if the GAF image is "predictable" or "unpredictable"
    cnn_model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = cnn_model(gaf_image)
        _, cnn_prediction = torch.max(output.data, 1)

    return cnn_prediction, output
#%%
# Train 50 models on
models=[]
for x in range(0,5):
    cnn_model_, best_accuracy = trainCNN()
    if best_accuracy > 0.7:
        models.append(cnn_model_)

cnn_model = models[0]
#%%
# Evaluate 50 models on test set
predictions_list = []
for cnn_model in models:
    predictions_cnn = []
    for i in range(len(X_test)):
        feature_window = X_test[i]  # Extract the feature window for the i-th sample
        prediction, output = predict_cnn(cnn_model, feature_window)
        predictions_cnn.append(prediction.item())
    predictions_list.append(predictions_cnn)
#%%
# save for later use and analiz
mx = np.array(predictions_list)
mx_ds = pd.DataFrame(mx)
mx_ds.to_csv('predictions.csv', index=False)
predictions_mx = pd.read_csv('predictions.csv').to_numpy().squeeze().transpose()
y_test_ds = pd.DataFrame(y_test, columns=['class'])
y_test_ds.to_csv('y_test.csv')
true_labels = pd.read_csv('y_test.csv')['class'].to_numpy()#.squeeze().squeeze().squeeze()
#%%
# Step 1: Calculate Accuracy for Each Model
individual_accuracies = np.mean(predictions_mx == true_labels[:, None], axis=0)
print("Individual Model Accuracies:", individual_accuracies)
# Step 2: Calculate Majority Vote
majority_vote = np.sum(predictions_mx, axis=1) > (predictions_mx.shape[1] / 2)

# Step 3: Calculate Majority Vote Accuracy
majority_vote_accuracy = np.mean(majority_vote == true_labels)
print("Majority Vote Accuracy:", majority_vote_accuracy)

# Step 4: Agreement vs. Correctness
agreement_percentage = np.sum(predictions_mx == true_labels[:, None], axis=1) / predictions_mx.shape[1] * 100
correct_predictions = majority_vote == true_labels

# Analyze agreement levels for correct and incorrect predictions
correct_agreement = agreement_percentage[correct_predictions]
incorrect_agreement = agreement_percentage[~correct_predictions]

print("Average Agreement for Correct Predictions:", np.mean(correct_agreement))
print("Average Agreement for Incorrect Predictions:", np.mean(incorrect_agreement))
#%%
threshold = 60  # Only make predictions if agreement is above this percentage

filtered_predictions = []
for i, agreement in enumerate(agreement_percentage):
    if agreement > threshold:
        filtered_predictions.append(majority_vote[i])
    else:
        filtered_predictions.append("No Prediction")

# Calculate accuracy for cases where predictions were made
valid_predictions = [pred for pred in filtered_predictions if pred != "No Prediction"]
valid_labels = [true_labels[i] for i, pred in enumerate(filtered_predictions) if pred != "No Prediction"]

accuracy_with_threshold = np.mean(np.array(valid_predictions) == np.array(valid_labels))
print(f'Accuracy with Threshold-Based Predictions: {accuracy_with_threshold:.2f}%')
#%%
threshold = 50  # Start with a lower threshold and adjust

filtered_predictions = []
valid_labels = []
for i, agreement in enumerate(agreement_percentage):
    if agreement > threshold:
        filtered_predictions.append(majority_vote[i])
        valid_labels.append(true_labels[i])
    else:
        filtered_predictions.append(None)  # or "No Prediction"

# Calculate accuracy for cases where predictions were made
valid_predictions = [pred for pred in filtered_predictions if pred is not None]

if len(valid_predictions) > 0:
    accuracy_with_threshold = np.mean(np.array(valid_predictions) == np.array(valid_labels))
    print(f'Accuracy with Threshold-Based Predictions: {accuracy_with_threshold:.2f}%')
else:
    print("No valid predictions were made based on the current threshold.")
print('np.shape(valid_predictions)', np.shape(valid_predictions))
#%%
# Print some sample predictions and their corresponding true labels
for i in range(10):  # Adjust range as needed
    print(f"Prediction: {valid_predictions[i]}, True Label: {valid_labels[i]}")

# Recheck the majority vote logic without threshold
majority_vote_accuracy = np.mean(majority_vote == true_labels)
print(f"Majority Vote Accuracy (without threshold): {majority_vote_accuracy:.2f}%")

# Compare agreement percentages with correctness
correctness_comparison = list(zip(majority_vote, true_labels, agreement_percentage))

for pred, true, agree in correctness_comparison[:10]:  # Check the first 10 for inspection
    print(f"Predicted: {pred}, True: {true}, Agreement: {agree:.2f}%")
#%%
# Check class distribution in the true labels
unique, counts = np.unique(true_labels, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Class Distribution:", class_distribution)

# Example output might be something like:
# Class Distribution: {0: 1000, 1: 3000}
#%%
# Step 6: Testing on the Test Set
# def predict_with_combined_model(cnn_model, svm_model, feature_window):
#     # No need to scale the feature_window again since it was done during data preparation
#
#     # Generate GAF image from the feature window
#     gaf = GramianAngularField(image_size=feature_window_size, method='summation')
#     gaf_image = gaf.fit_transform(feature_window.reshape(1, -1))
#
#     # Prepare the GAF image for the CNN
#     gaf_image = torch.tensor(gaf_image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension only
#
#
#     # Use CNN to predict if the GAF image is "predictable" or "unpredictable"
#     cnn_model.eval()  # Set the model to evaluation mode
#     with torch.no_grad():
#         output = cnn_model(gaf_image)
#         _, cnn_prediction = torch.max(output.data, 1)
#
#     if cnn_prediction.item() == 1:  # If CNN predicts "predictable"
#         # Use the SVM model to predict "up" or "down"
#         svm_prediction = svm_model.predict(feature_window.reshape(1, -1))
#         return "up" if svm_prediction[0] == 1 else "down"
#     else:
#         return "unpredictable"
#%%
# # Loop through each sample in the test set
# test_predictions = []
# cnn_predictable_count = 0
# correct_predictable_predictions = 0
#
# for i in range(len(X_test)):
#     feature_window = X_test[i]  # Extract the feature window for the i-th sample
#     prediction = predict_with_combined_model(cnn_model, svm_model, feature_window)
#
#     if prediction != "unpredictable":  # Only consider cases where the CNN predicted "predictable"
#         cnn_predictable_count += 1
#         test_predictions.append(prediction)
#
#         # Check if the prediction is correct
#         if (prediction == "up" and y_test[i] == 1) or (prediction == "down" and y_test[i] == 0):
#             correct_predictable_predictions += 1
#
# # Calculate accuracy only on the "predictable" cases
# if cnn_predictable_count > 0:
#     test_accuracy = correct_predictable_predictions / cnn_predictable_count * 100
#     print(f'Test Set Accuracy on "Predictable" Cases: {test_accuracy:.2f}%')
# else:
#     print("No 'Predictable' cases identified by the CNN.")
#%%
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
#%%
print("mx shape:", mx.shape)
