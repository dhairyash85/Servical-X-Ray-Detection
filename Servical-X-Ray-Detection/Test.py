from keras.models import load_model
import os
import pandas as pd
import pydicom
import numpy as np
model=load_model('trained_model.h5')
test = pd.read_csv('test.csv')
def load_dicom_images_from_folder(folder_path, metadata):
    images = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.dcm'):
            dcm = pydicom.dcmread(os.path.join(folder_path, file_name))
            image = dcm.pixel_array.astype(float) / np.max(dcm.pixel_array)  # Normalize pixel values
            images.append(image)
    # Return images
    return np.array(images)

train_images = []
for folder_name in test['']:
    folder_path = os.path.join('train', 'images', folder_name)
    images = load_dicom_images_from_folder(folder_path, train_metadata[train_metadata['StudyInstanceUID'] == folder_name])
    train_images.append(images)