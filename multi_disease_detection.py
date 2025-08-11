import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import os
from torchvision import transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
import numpy as np
import time
import ast

# =============================================================================
# STEP 1: CREATE THE DATASET (FOR MULTIPLE DISEASES)
# =============================================================================
class MultiDiseaseDataset(Dataset):
    def __init__(self, csv_file, image_dir="Training Images", transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.disease_columns = ['D', 'G', 'C', 'A', 'H', 'M', 'O'] #Disease labels
        print(f"Successfully loaded {len(self.df)} eye samples for multi-disease detection.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_filename = row['filename']
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        target_string = row['target']
        
        target_list = ast.literal_eval(target_string) #Makes the strin of diseases into a list
        
        disease_labels = target_list[1:]
        
        labels = torch.tensor(disease_labels, dtype=torch.float32) #Converts the list of diseases to a PyTorch tensor

        return image, labels

# =============================================================================
# STEP 2: PREPARE DATA LOADERS
# =============================================================================
def prepare_data():
    IMAGE_SIZE = 224

    #Shapes the image for the model
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = MultiDiseaseDataset(
        csv_file='full_df.csv',
        image_dir='Training Images',
        transform=train_transforms
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    return train_loader, val_loader, full_dataset.disease_columns

# =============================================================================
# STEP 3: CREATE THE NEURAL NETWORK MODEL
# =============================================================================
class MultiDiseaseClassifier(nn.Module): #Neural network creation for the 7 diseases
    def __init__(self, num_diseases=7):
        super(MultiDiseaseClassifier, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        #Freezes the intial layers of the resnet18 model and leaves the last 2 for training
        for name, param in self.backbone.named_parameters(): 
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        num_features = self.backbone.fc.in_features
    
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5), #Randomly drops 50% of the neurons during each batch to help combat overfitting
            nn.Linear(256, num_diseases)
        )

    def forward(self, x):
        return self.backbone(x)

# =============================================================================
# STEP 4: TRAINING FUNCTION
# =============================================================================
def train_model(model, train_loader, val_loader, disease_names, num_epochs=10):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    history = {'train_loss': [], 'val_loss': [], 'val_exact_accuracy': []} 

    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        #Training Phase
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)

        #Validation Phase
        model.eval()
        running_val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        #Calculates training and validation loss
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculates the exact accuracy of the 7 predicted diseases and the actual 7 disease labels
        exact_accuracy = np.mean(np.all(all_preds == all_labels, axis=1))
        history['val_exact_accuracy'].append(exact_accuracy)

        # Calculates the time taken during each epoch
        epoch_duration = time.time() - epoch_start_time

        print(f"\nEpoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Exact Match Accuracy: {exact_accuracy:.4f} |"
              f"Time: {epoch_duration:.2f}s")

        #Accuracy for predicting each disease
        print("--- Per-Disease Validation Accuracy ---")
        for i, name in enumerate(disease_names):
            disease_acc = accuracy_score(all_labels[:, i], all_preds[:, i])
            print(f"  {name}: {disease_acc:.4f}")
        
        scheduler.step(epoch_val_loss)
        
    print("\nTraining finished. Generating plots...")
    plt.figure(figsize=(12, 5))

    # Plots train loss and validation loss over number of epochs
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plots the exact match accuracy over number of epochs
    plt.subplot(1, 2, 2)
    plt.plot(history['val_exact_accuracy'], label='Exact Match Accuracy', color='orange')
    plt.title('Exact Match Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return model, history

# =============================================================================
# STEP 5: PREDICTION FUNCTION
# =============================================================================
def predict_diseases(model, image_path, disease_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs).squeeze().numpy()
        predictions = (probabilities > 0.5).astype(int)

    results = {}
    print(f"\n--- Prediction Results for: {image_path} ---")
    for i, name in enumerate(disease_names):
        status = "Detected" if predictions[i] == 1 else "Not Detected"
        print(f"  {name}: {status} (Confidence: {probabilities[i]:.4f})")
        results[name] = {"status": status, "confidence": probabilities[i]}
    
    return results

# =============================================================================
# STEP 6: MAIN TRAINING PIPELINE
# =============================================================================
def main():
    print("=== Multi-Disease Classification Model Training ===")
    
    # Step 1: Prepare data
    print("\n1. Preparing data...")
    train_loader, val_loader, disease_names = prepare_data()
    
    # Step 2: Create model
    print("\n2. Creating model...")
    model = MultiDiseaseClassifier(num_diseases=len(disease_names))

    # Step 3: Train model
    print("\n3. Starting training...")
    trained_model, history = train_model(model, train_loader, val_loader, disease_names, num_epochs=10)
    
    # Step 4: Save the trained model
    print("\n4. Saving model...")
    torch.save(trained_model.state_dict(), 'multi_disease_model.pth')
    print("Model saved as 'multi_disease_model.pth'")
    
    # Step 5: Test on a new image
    print("\n5. Testing on a new image...")
    try:
        predict_diseases(trained_model, 'test_image.jpg', disease_names)
    except FileNotFoundError:
        print("Test image not found. You can use the 'predict_diseases' function later.")
    
    print("Training complete!")

if __name__ == "__main__":
    main() #Run the model