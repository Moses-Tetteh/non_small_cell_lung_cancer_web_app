Optimise these codes for me and save the best model for me in with the title resnet18_lung_cancer_classifier.pth
I also get an error when i try to call see the accuracy of the model in website Prediction Result
adenocarcinoma Confidence: N/A


import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets.folder import has_file_allowed_extension
import shutil
import os # Import the os module
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from timeit import default_timer as timer
from google.colab import drive
from torch.optim import lr_scheduler # Import lr_scheduler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc # Import classification_report, roc_curve, auc
import seaborn as sns # Import seaborn for heatmap visualization
from sklearn.preprocessing import label_binarize # Import label_binarize for ROC curve

# Mount Google Drive to access your dataset
drive.mount('/content/drive')

# Device configuration: Use CUDA (GPU) if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load pre-trained ResNet18 model with ImageNet weights
# This model has been pre-trained on a large dataset (ImageNet) and provides
# good feature extraction capabilities for transfer learning.
resnet18_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# Move the model to the selected device (GPU or CPU)
resnet18_model = resnet18_model.to(device)

# ✅ STEP 1: Define paths for your dataset
# This is the base directory where your 'train', 'test', 'valid' folders are located.
base_dir = "/content/drive/MyDrive/archive (1)"
sub_dirs = ['train', 'test', 'valid']
# Define valid image file extensions to ensure only image files are processed.
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

# NOTE: The clean_and_rename_classes function has been removed as per your request,
# assuming you have already manually organized and renamed your class folders.

# ✅ STEP 3: (Skipping Step 2 as clean_and_rename_classes is removed)
# The dataset folders are assumed to be correctly structured with class names as subfolder names.
print("Assuming class folders are already correctly named and structured.")

# ✅ STEP 4: Define data transforms
# Data augmentation is crucial for training to help the model generalize better
# by introducing variability in the training data.
train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)), # Resize images to 224x224, as expected by ResNet
    transforms.RandomHorizontalFlip(p=0.5), # Randomly flip images horizontally
    transforms.RandomRotation(15), # Randomly rotate images by up to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Randomly change color properties
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0), # Randomly translate and scale
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)), # Randomly crop and resize
    transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor
    # Normalize image tensors with ImageNet mean and standard deviation.
    # This is standard practice when using pre-trained models on ImageNet.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# For validation and test data, we only resize and normalize, without augmentation,
# to ensure consistent evaluation.
test_val_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ STEP 5: Load datasets using ImageFolder
# ImageFolder automatically infers class labels from subfolder names.
print("Loading datasets...")
train_data = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(base_dir, 'test'), transform=test_val_transform)
validation_data = datasets.ImageFolder(os.path.join(base_dir, 'valid'), transform=test_val_transform)
print("Datasets loaded.")

# ✅ STEP 6: Print class names and dataset sizes to verify
# These are automatically populated by ImageFolder based on your cleaned folder structure.
class_names = train_data.classes
class_dict = train_data.class_to_idx

print("\nClasses (from dataset):", class_names)
print("Class to index mapping (from dataset):", class_dict)
print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Validation samples: {len(validation_data)}")

# Verify a random image and its label from the training dataset
random_idx = torch.randint(0, len(train_data), size=(1,)).item()
img, label = train_data[random_idx] # ImageFolder returns (image, label) tuple directly

print(f"\nRandom training image details:")
print(f"Image tensor shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label (index): {label}")
print(f"Label datatype: {type(label)}")
print(f"Class name: {class_names[label]}")

# Display a random training image
img_permute = img.permute(1, 2, 0) # Permute from (C, H, W) to (H, W, C) for matplotlib
# Denormalize the image for correct display, as it was normalized by the transform
img_display = img_permute * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
img_display = torch.clamp(img_display, 0, 1) # Clamp values to [0, 1] to ensure valid image data

plt.figure(figsize=(8, 6))
plt.imshow(img_display)
plt.axis("off") # Hide axes for a cleaner look
plt.title(f"Class: {class_names[label]}", fontsize=14)
plt.show()

# Define Batch Size and DataLoaders
BATCH_SIZE = 32
# DataLoaders efficiently load data in batches for training.
# `num_workers=os.cpu_count()` utilizes all available CPU cores for data loading,
# which can speed up training, especially on CPU-bound operations.
# `pin_memory=True` helps in faster data transfer to GPU.
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=os.cpu_count(),
                              shuffle=True, # Shuffle training data for better generalization
                              pin_memory=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             num_workers=os.cpu_count(),
                             shuffle=False, # No need to shuffle test data
                             pin_memory=True)

validation_dataloader = DataLoader(dataset=validation_data,
                                  batch_size=BATCH_SIZE,
                                  num_workers=os.cpu_count(),
                                  shuffle=False, # No need to shuffle validation data
                                  pin_memory=True)

print(f"\nTrain Dataloader batches: {len(train_dataloader)}")
print(f"Test Dataloader batches: {len(test_dataloader)}")
print(f"Validation Dataloader batches: {len(validation_dataloader)}")

# Modify the final classification layer (Fully Connected layer) of ResNet18
# The original ResNet18 was designed for 1000 ImageNet classes. We need to
# adapt its final layer for our specific number of lung cancer classes.
# `num_ftrs` gets the number of input features to the original FC layer (512 for ResNet18).
num_ftrs = resnet18_model.fc.in_features

# Redefine the final classification layer as a sequential block.
# This allows for adding more layers, activations, and dropout for better performance.
resnet18_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512), # First linear layer
    nn.ReLU(), # ReLU activation for non-linearity
    nn.Dropout(0.3), # Dropout for regularization to prevent overfitting
    nn.Linear(512, 256), # Second linear layer
    nn.ReLU(), # ReLU activation
    nn.Dropout(0.3), # Dropout
    nn.Linear(256, len(class_names)) # Final linear layer, output features must match the number of classes
)
# Ensure the new FC layer is also moved to the correct device
resnet18_model.fc = resnet18_model.fc.to(device)

# The `resnet18_model.conv1` line from your original code is usually not needed
# unless you are changing the number of input channels (e.g., grayscale images).
# For standard RGB images (3 channels), the default `conv1` is already suitable.
# resnet18_model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


# Define Loss function and Optimizer
loss_fn = nn.CrossEntropyLoss() # CrossEntropyLoss is suitable for multi-class classification

# Optimizer: AdamW is often preferred for its adaptive learning rates and decoupled weight decay,
# which can lead to better generalization compared to Adam or SGD in some cases.
# A lower learning rate (e.g., 0.0001) is often good for fine-tuning pre-trained models.
optimizer = torch.optim.AdamW(resnet18_model.parameters(), lr=0.0001, weight_decay=1e-4)

# Learning Rate Scheduler: ReduceLROnPlateau reduces the learning rate when a metric
# (here, test_loss) has stopped improving. This helps the model converge better.
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Model summary: Provides a detailed overview of the model architecture,
# output shapes, and number of parameters.
from torchinfo import summary
# Input size should match the size of images after transforms (batch_size, channels, height, width)
summary(resnet18_model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"], row_settings=["var_names"], verbose=0)


# Training Step: Performs one forward and backward pass for a single epoch on the training data.
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    model.train() # Set model to training mode
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # Move data to the specified device

        y_pred = model(X) # Forward pass

        loss = loss_fn(y_pred, y) # Calculate loss
        train_loss += loss.item() # Accumulate loss

        optimizer.zero_grad() # Zero gradients before backward pass
        loss.backward() # Backward pass: compute gradients
        optimizer.step() # Update model parameters

        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Calculate average loss and accuracy for the epoch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

# Test/Validation Step: Evaluates the model on the test/validation data.
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    model.eval() # Set model to evaluation mode (disables dropout, batch norm updates)
    test_loss, test_acc = 0, 0

    with torch.inference_mode(): # Disable gradient calculations for efficiency
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X) # Forward pass

            loss = loss_fn(test_pred_logits, y) # Calculate loss
            test_loss += loss.item() # Accumulate loss

            # Calculate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Calculate average loss and accuracy for the epoch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# Main Training Function: Orchestrates the training and evaluation process over multiple epochs.
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device: torch.device = device,
          scheduler=None,
          early_stopping_patience: int = 7): # Added early stopping patience

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    best_test_loss = float('inf') # Initialize best test loss to infinity
    patience_counter = 0 # Counter for early stopping patience
    best_model_state = None # To store the state_dict of the best model

    for epoch in tqdm(range(epochs)): # tqdm provides a nice progress bar
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        # Print epoch-wise results
        print(f"Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

        # Update the learning rate scheduler based on the test loss
        if scheduler:
            scheduler.step(test_loss)

        # Early stopping logic
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0 # Reset patience since test loss improved
            best_model_state = model.state_dict() # Save the best model state
            print(f"  --> Test loss improved. Saving model state. Best test loss: {best_test_loss:.4f}")
        else:
            patience_counter += 1 # Increment patience if test loss did not improve
            print(f"  --> Test loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs. Test loss did not improve for {early_stopping_patience} consecutive epochs.")
                break # Stop training

        # Store results for plotting/analysis later
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Load the best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model state based on test loss.")

    return results

# Set number of epochs for training
NUM_EPOCHS = 50 # Increased max epochs, but early stopping will manage actual epochs

# Start training
print("\nStarting training...")
start_time = timer() # Record start time

model_0_results = train(model=resnet18_model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS,
                        device=device,
                        scheduler=scheduler,
                        early_stopping_patience=10) # Set early stopping patience

end_time = timer() # Record end time
print(f"Total training time: {end_time - start_time:.3f} seconds")

# Save the trained model's state dictionary
# It's good practice to save the state_dict (model parameters) rather than the whole model,
# as it's more flexible for loading later.
model_save_path = "resnet18_lung_cancer_classifier.pth" # Corrected and descriptive filename
torch.save(resnet18_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# --- Prediction and Visualization ---
print("\nPerforming predictions and visualization...")

# The `resnet18_model` now holds the best weights due to early stopping.

imag_samples = []
true_labels = []

# Collect 16 random images and their true labels from the test dataset for visualization
num_samples_to_plot = 16
for _ in range(num_samples_to_plot):
    random_idx = torch.randint(0, len(test_data), size=(1,)).item()
    img, label = test_data[random_idx] # img is already a tensor after test_val_transform
    imag_samples.append(img)
    true_labels.append(label)

# Define the prediction function
# This function takes an image tensor, the trained model, and the device,
# and returns the predicted class index and the confidence of the prediction.
def predict_image(image_tensor, model, device=device):
    # The input `image_tensor` is expected to be already transformed (C, H, W).
    # Unsqueeze to add a batch dimension (1, C, H, W) and move to the device.
    image_input = image_tensor.unsqueeze(0).to(device)

    model.eval() # Set model to evaluation mode
    with torch.inference_mode(): # Disable gradient calculations for faster inference
        outputs = model(image_input) # Get raw logits from the model
        probabilities = torch.softmax(outputs, dim=1) # Convert logits to probabilities
        predicted_class_idx = probabilities.argmax(dim=1).item() # Get the index of the highest probability
        confidence = probabilities.max().item() # Get the maximum probability (confidence)

    return predicted_class_idx, confidence

plt.figure(figsize=(15, 15)) # Adjusted figure size for better display of 16 images

for i in range(num_samples_to_plot):
    image_tensor = imag_samples[i]
    true_class_idx = true_labels[i]
    true_class_name = class_names[true_class_idx]

    # Predict the image class using the trained model
    predicted_class_idx, confidence = predict_image(image_tensor, resnet18_model, device)
    predicted_class_name = class_names[predicted_class_idx]

    # Prepare image for display:
    # 1. Move the image_tensor to the device before permuting and denormalizing.
    # This resolves the RuntimeError by ensuring all tensors are on the same device.
    image_tensor_on_device = image_tensor.to(device)
    # 2. Permute dimensions from (C, H, W) to (H, W, C) for matplotlib.
    img_display = image_tensor_on_device.permute(1, 2, 0)
    # 3. Denormalize the image to bring pixel values back to a displayable range [0, 1].
    # This reverses the normalization applied in the transforms.
    img_display = img_display * torch.tensor([0.229, 0.224, 0.225], device=device) + torch.tensor([0.485, 0.456, 0.406], device=device)
    # 4. Clamp values to [0, 1] to handle any floating point inaccuracies during denormalization.
    # 5. Move tensor to CPU and convert to NumPy array for matplotlib.
    img_display = torch.clamp(img_display, 0, 1).cpu().numpy()

    # Plot the image with true and predicted labels
    plt.subplot(4, 4, i + 1) # Arrange plots in a 4x4 grid
    plt.imshow(img_display)

    # Set title color based on prediction accuracy
    title_color = "g" if true_class_name == predicted_class_name else "r"
    plt.title(f"True: {true_class_name}\nPred: {predicted_class_name} ({confidence*100:.1f}%)",
              fontsize=10, color=title_color)
    plt.axis('off') # Hide axes

plt.tight_layout() # Adjust subplot parameters for a tight layout
plt.show()

# --- Confusion Matrix and Heatmap Visualization ---
print("\nGenerating Confusion Matrix Heatmap...")

# Collect all true labels and predicted labels from the test set
all_true_labels = []
all_predicted_labels = []
all_probabilities = [] # Added for ROC curve

resnet18_model.eval() # Set model to evaluation mode
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Predicting on test set"):
        X, y = X.to(device), y.to(device)
        outputs = resnet18_model(X)
        predicted_labels = outputs.argmax(dim=1)
        probabilities = torch.softmax(outputs, dim=1) # Get probabilities

        all_true_labels.extend(y.cpu().numpy())
        all_predicted_labels.extend(predicted_labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy()) # Store probabilities

# Compute the confusion matrix
cm = confusion_matrix(all_true_labels, all_predicted_labels)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print("Confusion Matrix Heatmap generated successfully.")

# --- Classification Report ---
print("\nGenerating Classification Report...")
# This provides precision, recall, and F1-score for each class.
print(classification_report(all_true_labels, all_predicted_labels, target_names=class_names))
print("Classification Report generated successfully.")


# --- Loss and Accuracy Curves over Epochs ---
print("\nGenerating Loss and Accuracy Curves over Epochs...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(model_0_results['train_loss'], label='Train Loss')
plt.plot(model_0_results['test_loss'], label='Test Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(model_0_results['train_acc'], label='Train Accuracy')
plt.plot(model_0_results['test_acc'], label='Test Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("Loss and Accuracy Curves generated successfully.")


# --- ROC Curve and AUC ---
print("\nGenerating ROC Curve and AUC...")
all_probabilities = np.array(all_probabilities)
all_true_labels_binarized = label_binarize(all_true_labels, classes=range(len(class_names)))

plt.figure(figsize=(10, 8))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(all_true_labels_binarized[:, i], all_probabilities[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
print("ROC Curve and AUC generated successfully.")