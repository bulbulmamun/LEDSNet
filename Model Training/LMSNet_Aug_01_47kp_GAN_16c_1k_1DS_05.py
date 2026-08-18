#######################################################################################
## Lightweight MobSqu ECG classifier  full script with GAN-based augmentation
#######################################################################################

#######################################################################################
# 0.  IMPORTS
#######################################################################################
import sys, platform, os, random, time, itertools
import numpy as np
import pandas as pd
import h5py, scipy.io
import sklearn as sk
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd          # NEW  for gradient penalty
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
# (SMOTE import removed)

#######################################################################################
# 1.  Reproducibility & device
#######################################################################################
torch.manual_seed(42);  np.random.seed(42);  random.seed(42)

index_combination = 5
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device  = torch.device("mps" if has_mps else "cuda" if has_gpu else "cpu")

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Target device  : {device}")

#######################################################################################
# 2.  Naming, training parameters
#######################################################################################
num_epochs = 1000
model_Num  = f'LMSNet_{index_combination:02}_47kp_16c_GAN_Ep_1k_1DS_{num_epochs}'
mod_name   = f'{model_Num}.pth'
mod_weight = f'{model_Num}_parameters.pth'
mod_weight_last = f'{model_Num}_parameters_last.pth'
print(mod_name);  print(mod_weight);  print(mod_weight_last)

#######################################################################################
# 3.  LOAD DATASETS  (unchanged)
#######################################################################################
directory           = '/scratch/user/uqabulbu/Data/'
# file_name_MITBIHSVA = 'Data_AAMB/AR/1.MITBIHSVA_Filtered_Segmented_2CH_128Hz.h5'
file_name_MITBIHAR  = 'Data_AAMB/AR/2.MITBIHAR_Filtered_Segmented_1CH_128HZ.h5'
# file_path_MITBIHSVA = directory + file_name_MITBIHSVA
file_path_MITBIHAR  = directory + file_name_MITBIHAR

# with h5py.File(file_path_MITBIHSVA, 'r') as h5f:
#     ecg_signals_1 = h5f['ECG_Signals'][:]
#     ecg_labels_1  = h5f['ECG_Labels'][:]
# ecg_signals_1 = ecg_signals_1[:, 0, :]  # keep Lead-II

with h5py.File(file_path_MITBIHAR, 'r') as h5f:
    ecg_signals_2 = h5f['ECG_Signals'][:]
    ecg_labels_2  = h5f['ECG_Labels'][:]

# combined_signals = np.concatenate((ecg_signals_1, ecg_signals_2), axis=0)
# combined_labels  = np.concatenate((ecg_labels_1,  ecg_labels_2),  axis=0).squeeze()

combined_signals = ecg_signals_2
combined_labels  = ecg_labels_2.squeeze()

print("Combined data shape :", combined_signals.shape)
print("Combined label shape:", combined_labels.shape)

#######################################################################################
# 4.  Train/val/test split  (unchanged)
#######################################################################################
data_tensor   = torch.from_numpy(combined_signals).float().unsqueeze(1)  # (N,1,77)
labels_tensor = torch.from_numpy(combined_labels).long()

data_np   = data_tensor.squeeze(1).numpy()  # (N,77)
labels_np = labels_tensor.numpy()

train_val_data, test_data, train_val_labels, test_labels = train_test_split(
    data_np, labels_np, test_size=0.30, stratify=labels_np, random_state=42)

train_data, val_data, train_labels, val_labels = train_test_split(
    train_val_data, train_val_labels, test_size=0.10,
    stratify=train_val_labels, random_state=42)

train_data_tensor = torch.from_numpy(train_data).float().unsqueeze(1)
val_data_tensor   = torch.from_numpy(val_data).float().unsqueeze(1)
test_data_tensor  = torch.from_numpy(test_data).float().unsqueeze(1)

train_labels_tensor = torch.from_numpy(train_labels).long()
val_labels_tensor   = torch.from_numpy(val_labels).long()
test_labels_tensor  = torch.from_numpy(test_labels).long()

train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
val_dataset   = TensorDataset(val_data_tensor,   val_labels_tensor)
test_dataset  = TensorDataset(test_data_tensor,  test_labels_tensor)

batch_size  = 4096
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

#######################################################################################
# 5.  **GAN-BASED AUGMENTATION**  (replaces former SMOTE block)
#######################################################################################
print("Before GAN class distribution:",
      np.bincount(train_labels_tensor.view(-1).numpy().astype(int)))

noise_dim            = 100
gan_epochs_per_class = 50
n_critic             = 5
lmda_gp                 = 10.0
min_samples_per_class = 47102      # same target as before

class Gen(nn.Module):
    def __init__(self, noise_dim=100, num_classes=16, out_dim=77):
        super().__init__()
        self.embed = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 128), nn.ReLU(True),
            nn.Linear(128, 256), nn.ReLU(True),
            nn.Linear(256, out_dim)
        )
    def forward(self, z, y):
        y_emb = self.embed(y)
        return self.net(torch.cat([z, y_emb], dim=1))

class Critic(nn.Module):
    def __init__(self, num_classes=16, in_dim=77):
        super().__init__()
        self.embed = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(in_dim + num_classes, 256), nn.LeakyReLU(0.2, True),
            nn.Linear(256, 128), nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1)
        )
    def forward(self, x, y):
        y_emb = self.embed(y)
        return self.net(torch.cat([x, y_emb], dim=1))

def gradient_penalty(critic, real, fake, labels):
    alpha = torch.rand(real.size(0), 1, device=device)
    inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = critic(inter, labels)
    grad  = autograd.grad(outputs=score, inputs=inter,
                          grad_outputs=torch.ones_like(score),
                          create_graph=True, retain_graph=True,
                          only_inputs=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()

#  train a conditional WGAN-GP for every scarce class 
train_data_sm = train_data_tensor.squeeze(1).numpy()   # (N,77)
train_labels_sm = train_labels_tensor.numpy()

synthetic_batches, synthetic_labels = [], []

for cls in np.unique(train_labels_sm):
    cls_idx   = train_labels_sm == cls
    cls_count = cls_idx.sum()
    if cls_count >= min_samples_per_class:
        continue                        # already plenty

    real_cls = torch.tensor(train_data_sm[cls_idx],
                            dtype=torch.float32, device=device)
    lbl_cls  = torch.tensor(train_labels_sm[cls_idx],
                            dtype=torch.long,   device=device)

    G = Gen(noise_dim, 16, 77).to(device)
    D = Critic(16, 77).to(device)
    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

    loader_cls = DataLoader(TensorDataset(real_cls, lbl_cls),
                            batch_size=512, shuffle=True, drop_last=False)

    for _ in range(gan_epochs_per_class):
        for x_real, y_real in loader_cls:
            #  critic 
            for _ in range(n_critic):
                z = torch.randn(x_real.size(0), noise_dim, device=device)
                x_fake = G(z, y_real).detach()
                loss_D = D(x_fake, y_real).mean() - D(x_real, y_real).mean()
                gp = gradient_penalty(D, x_real, x_fake, y_real)
                loss_D = loss_D + lmda_gp * gp
                opt_D.zero_grad();  loss_D.backward();  opt_D.step()

            #  generator 
            z = torch.randn(x_real.size(0), noise_dim, device=device)
            x_fake = G(z, y_real)
            loss_G = -D(x_fake, y_real).mean()
            opt_G.zero_grad();  loss_G.backward();  opt_G.step()

    # generate until the class is topped-up
    needed = min_samples_per_class - cls_count
    with torch.no_grad():
        while needed > 0:
            gen_batch = min(needed, 4096)
            z   = torch.randn(gen_batch, noise_dim, device=device)
            lab = torch.full((gen_batch,), cls, dtype=torch.long, device=device)
            fake = G(z, lab).cpu().numpy()
            synthetic_batches.append(fake)
            synthetic_labels.append(np.full(gen_batch, cls, dtype=train_labels_sm.dtype))
            needed -= gen_batch

#  merge synthetic & real, rebuild DataLoader 
if synthetic_batches:
    synth_data   = np.vstack(synthetic_batches)
    synth_labels = np.hstack(synthetic_labels)

    new_train_data   = np.vstack([train_data_sm, synth_data])
    new_train_labels = np.hstack([train_labels_sm, synth_labels])

    new_train_data, new_train_labels = shuffle(new_train_data,
                                               new_train_labels,
                                               random_state=42)

    train_data_tensor   = torch.tensor(new_train_data).float().unsqueeze(1)
    train_labels_tensor = torch.tensor(new_train_labels).long()

    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

else:
    new_train_labels = train_labels_sm   # nothing changed

print("After  GAN  class distribution:",
      np.bincount(new_train_labels.astype(int)))

#######################################################################################
# 6.  CLASS WEIGHTS  (unchanged logic, new data)
#######################################################################################
class_counts = np.bincount(new_train_labels.astype(int))
total_samples = len(new_train_labels)
class_weights = total_samples / (len(class_counts) * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("Class counts :", class_counts)
print("Class weights:", class_weights_tensor)

#######################################################################################
# 7.  MODEL DEFINITION
#######################################################################################
# 3. Model Definition

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        # Pointwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModule, self).__init__()
        self.squeeze = nn.Sequential(
            nn.Conv1d(in_channels, squeeze_channels, kernel_size=1),
            nn.BatchNorm1d(squeeze_channels),
            nn.ReLU(inplace=True)
        )
        self.expand1x1 = nn.Sequential(
            nn.Conv1d(squeeze_channels, expand1x1_channels, kernel_size=1),
            nn.BatchNorm1d(expand1x1_channels),
            nn.ReLU(inplace=True)
        )
        self.expand3x3 = nn.Sequential(
            nn.Conv1d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(expand3x3_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1)

# Define a custom squeezing layer
class Squeeze(nn.Module):
    def forward(self, x):
        return torch.squeeze(x, -1)
    
class CombinedModel(nn.Module):
    def __init__(self, num_classes=16):
        super(CombinedModel, self).__init__()
        self.stages = nn.ModuleDict()

        # MobileNet stages
        self.stages['Stage 1'] = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.stages['Stage 2'] = DepthwiseSeparableConv1d(64, 32, kernel_size=5, stride=1, padding=1)
        self.stages['Stage 3'] = DepthwiseSeparableConv1d(32, 16, kernel_size=5, stride=4, padding=1)
        self.stages['Stage 4'] = DepthwiseSeparableConv1d(16, 16, kernel_size=5, stride=1, padding=1)

        # SqueezeNet stages
        self.stages['Stage 5'] = FireModule(16, 32, 16, 16)  # FireModule expects in_channels=128
        self.stages['Stage 6'] = FireModule(32, 64, 8, 8)  # FireModule expects in_channels=128
        #self.stages['Stage 5'] = FireModule(128, 16, 64, 64)  # Output channels after FireModule is 64+64=128
        #self.stages['Stage 6'] = FireModule(128, 32, 128, 128)  # Output channels: 128+128=256

        # Final layers
        self.stages['Stage 7'] = nn.AdaptiveAvgPool1d(1)
        self.stages['Stage 8'] = nn.Flatten()
        #self.stages['Stage 9'] = Squeeze()
        #self.stages['Stage 10'] = nn.Conv1d(16, num_classes, kernel_size=1)
        self.stages['Stage 9'] = nn.Linear(16, num_classes)  # Input features match the output channels of last FireModule

    def forward(self, x):
        for stage in self.stages.values():
            x = stage(x)
        return x


#############  NO CLASS WEIGHTS  #############
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss function, optimizer
model = CombinedModel(num_classes=16).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
print(model)


#############  TRAIN WITH CLASS WEIGHTS  #############
# Assume you have class weights or other parameters
#class_weights = class_weights.to(device)

# Set device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss function, optimizer
#model = CombinedModel(num_classes=16).to(device)
#criterion = nn.CrossEntropyLoss(weight=class_weights)
#optimizer = optim.Adam(model.parameters(), lr=0.01)
#print(model)



#############################################################
# 4. Training Loop
#############################################################

# CodeCarbon tracker -----------------------------------------------------------
from codecarbon import OfflineEmissionsTracker
tracker = OfflineEmissionsTracker(
    measure_power_secs=1,
    country_iso_code="AUS",
    project_name="FL-MITBIHAR",
    log_level="error")

# Start the timer
start_time = time.time()
t_wall0 = time.perf_counter()
tracker.start()
# ------------- everything after this gets measured ---------------------------#

# Initialize the minimum validation loss and maximum validation accuracy
min_avg_val_loss = 100  # Arbitrary high value to start
max_avg_val_accuracy = 0  # Initial value of max accuracy
best_model_epoch = 0  # Initial value of the epoch for the best model


# Training and validation loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss, total_correct, total_samples = 0, 0, 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(dim=1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Calculate batch accuracy
        batch_accuracy = 100.0 * (predicted == labels).sum().item() / labels.size(0)
        #if (i%100)==0:
        #  print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.2f}%")

    # Print epoch-level average loss and accuracy
    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = 100.0 * total_correct / total_samples
    print(f"End of Epoch {epoch+1}: Avg. Train Loss: {epoch_loss:.4f}, Avg. Train Accuracy: {epoch_accuracy:.2f}%")

    # Validation
    model.eval()  # Set the model to evaluation mode
    val_loss, val_correct, val_samples = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            val_correct += (predicted == labels).sum().item()
            val_samples += labels.size(0)

    # Print validation loss and accuracy
    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = 100.0 * val_correct / val_samples
    print(f"End of Epoch {epoch+1}: Avg. Val Loss: {avg_val_loss:.4f}, Avg. Val Accuracy: {avg_val_accuracy:.2f}%")
    
    # Check if the current model is better than the previous best
#    if avg_val_loss < min_avg_val_loss and avg_val_accuracy > max_avg_val_accuracy:
    if avg_val_accuracy > max_avg_val_accuracy:
        print(f"New best model found at epoch {epoch+1} with Val Loss: {avg_val_loss:.4f} and Val Accuracy: {avg_val_accuracy:.2f}%")
        min_avg_val_loss = avg_val_loss
        max_avg_val_accuracy = avg_val_accuracy
        best_model_epoch = epoch + 1

        # Save the best model
        model_path_1 = os.path.join(directory, 'Output_Models', mod_name)
        torch.save(model, model_path_1)

        # Save the best model's parameters
        model_path = os.path.join(directory, 'Output_Models', mod_weight)
        torch.save(model.state_dict(), model_path)

print("Training and Validation Done")
print("Epoch for the Best model:", best_model_epoch)



#############################################################
# 5. Testing at the last epoch model
#############################################################

# model.eval()
# test_loss, correct, test_samples = 0, 0, 0
# all_preds, all_labels = [], []

# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         test_loss += loss.item()
        
#         preds = outputs.argmax(dim=1)
#         correct += (preds == labels).sum().item()
#         test_samples += labels.size(0)

#         # Collect all predictions and actual labels
#         all_preds.extend(preds.view(-1).cpu().numpy())
#         all_labels.extend(labels.view(-1).cpu().numpy())
# print("Testing Done")

# # Calculate test loss and accuracy
# test_accuracy = 100.0 * correct / test_samples
# print(f"Test Loss: {test_loss / len(test_loader):.4f}")
# print(f"Test Accuracy: {test_accuracy:.2f}%")

# # Compute the confusion matrix
# cm = confusion_matrix(all_labels, all_preds)
# print("Confusion Matrix:")
# print(cm)

# # Generate a classification report
# report = classification_report(all_labels, all_preds)
# print("Classification Report:")
# print(report)


# # Save the last model's parameters
# model_path_last = os.path.join(directory, 'Output_Models', mod_weight_last)
# torch.save(model.state_dict(), model_path_last)



# -----------------------------------------------------------------------------#
# 13.  Stop CodeCarbon & print footprint                                       #
# -----------------------------------------------------------------------------#
emissions_g = tracker.stop()
end_time = time.time()
t_wall1 = time.perf_counter()

report      = tracker.final_emissions_data
energy_kwh  = report.energy_consumed
duration_s  = report.duration
avg_power_w = (energy_kwh * 1000) / (duration_s / 3600)

# Convert timestamps to human-readable format
start_time_str = datetime.fromtimestamp(start_time).strftime('%H:%M:%S')
end_time_str = datetime.fromtimestamp(end_time).strftime('%H:%M:%S')
elapsed_time = end_time - start_time

# Print the formatted start and end times and the elapsed time
print(f"Training and validation started at {start_time_str} and ended at {end_time_str} which in total takes {elapsed_time:.2f} seconds")


print("\n==============  TRAINING FOOTPRINT  ==============")
print(f"Total wall-time      : {t_wall1 - t_wall0:.4f} s")
print(f"Energy consumed      : {energy_kwh:.6f} kWh")
print(f"Average power        : {avg_power_w:.4f} W")
print(f"CO-eq emissions     : {emissions_g:.6f} g")
print("===================================================")







#############################################################
# Load the best model parameters before testing
best_model_path = os.path.join(directory, 'Output_Models', mod_weight)
model.load_state_dict(torch.load(best_model_path))



# CodeCarbon tracker -----------------------------------------------------------
tracker_test = OfflineEmissionsTracker(
    measure_power_secs=1,
    country_iso_code="AUS",
    project_name="FL-MITBIHAR",
    log_level="error")

test_wall0 = time.perf_counter()
tracker_test.start()
# ------------- everything after this gets measured ---------------------------#



# Testing loop
model.eval()
test_loss, correct, test_samples = 0, 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        test_samples += labels.size(0)

        # Collect all predictions and actual labels
        all_preds.extend(preds.view(-1).cpu().numpy())
        all_labels.extend(labels.view(-1).cpu().numpy())
print("Testing Done")

# Calculate test loss and accuracy
test_accuracy = 100.0 * correct / test_samples
print(f"Test Loss: {test_loss / len(test_loader):.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Compute the confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# Generate a classification report
report = classification_report(all_labels, all_preds)
print("Classification Report:")
print(report)

# -----------------------------------------------------------------------------#
# 13.  Stop CodeCarbon & print footprint                                       #
# -----------------------------------------------------------------------------#
emissions_g = tracker_test.stop()
test_wall1 = time.perf_counter()

report      = tracker_test.final_emissions_data
energy_kwh  = report.energy_consumed
duration_s  = report.duration
avg_power_w = (energy_kwh * 1000) / (duration_s / 3600)

print("\n==============  Testing FOOTPRINT  ==============")
print(f"Total wall-time      : {test_wall1 - test_wall0:.4f} s")
print(f"Energy consumed      : {energy_kwh:.6f} kWh")
print(f"Average power        : {avg_power_w:.4f} W")
print(f"CO-eq emissions     : {emissions_g:.6f} g")
print("===================================================")

#############################################################
# 7. Parameter Counting
#############################################################

from torchsummary import summary

# Assuming your model is on the correct device
summary(model, input_size=(1, 77))
