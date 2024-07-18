import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm  # Import tqdm for progress bar
from models.biometry_model import build_3d_cnn_with_2d_base

# Path to the dataset
dataset_path = '/content/drive/MyDrive/Fetal Tissue Annotation Challenge/feta_2.3/derivatives/biometry/'

# Load the biometry measurements from the TSV file
biometry_file_path = os.path.join(dataset_path, 'biometry.tsv')
biometry_df = pd.read_csv(biometry_file_path, sep='\t')

# Rename the 'Unnamed: 0' column to 'Subject'
biometry_df.rename(columns={'Unnamed: 0': 'Subject'}, inplace=True)

# Drop any unnamed columns that might have been included
biometry_df = biometry_df.loc[:, ~biometry_df.columns.str.contains('^Unnamed')]

print(biometry_df.columns)

# Prepare the dataset
X = []
y = []

# Iterate through each subject directory and process images
for subject_id in tqdm(biometry_df['Subject'], desc="Processing subjects"):
    subject_path = os.path.join(dataset_path, subject_id)

    t2_data_norm = load_and_preprocess_image(subject_path)

    if t2_data_norm is not None:
        t2_data_norm = np.repeat(t2_data_norm[..., np.newaxis], 3, axis=-1)
        X.append(t2_data_norm)
        subject_measurements = biometry_df[biometry_df['Subject'] == subject_id][['LCC', 'HV', 'bBIP', 'sBIP', 'TCD']].values[0]
        y.append(subject_measurements)

X = np.array(X)
y = np.array(y)

# Handle missing values by filling them with the mean of the column
y = np.where(np.isnan(y), np.nanmean(y, axis=0), y)

# Normalize the targets
scaler = StandardScaler()
y = scaler.fit_transform(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = build_3d_cnn_with_2d_base((128, 128, 128, 3))

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Set up callbacks
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
history = model.fit(X_train, y_train, epochs=2, batch_size=1, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping])

# Evaluate the model on the validation set
val_loss, val_mae = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}')
print(f'Validation MAE: {val_mae}')

# Load the best model
model.load_weights('best_model.h5')

# Save the final model
model.save('biometry_measurement_model.h5')
print("Final model saved to 'biometry_measurement_model.h5'")

# Plot training history
plt.figure(figsize=(12, 6))

# Plot MAE
plt.subplot(1, 2, 1)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
