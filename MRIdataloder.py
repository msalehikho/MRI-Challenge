import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pydicom
import cv2
import pandas as pd
from scipy.ndimage.interpolation import zoom

def position_finder(pos):
        pos = list(np.round(pos))
        if pos == [1.0 , 0.0 ,0.0 ,0.0, 0.0, -1.0]:
            return 'Coronal'
        elif pos == [0.0 , 1.0 ,0.0 ,0.0, 0.0, -1.0]:
            return 'Sagittal'
        elif pos == [1.0 , 0.0 ,0.0 ,0.0, 1.0, 0.0]:
            return 'Axial'
        else:
            print(pos)
            return 'None'
        
def Type_Axial_filter(path,label_path):
    Type = []
    Position = []
    df = pd.read_csv(label_path)
    for i in df['SeriesInstanceUID']:
        SerisPath = os.path.join(path,i)
        SamplePath = os.path.join(SerisPath,os.listdir(SerisPath)[0])
        Sample = pydicom.dcmread(SamplePath)
        Position.append(position_finder(Sample.ImageOrientationPatient))
        Type.append(Sample.SeriesDescription)
        
    df['Orientation']= Position
    df['Modality']= Type
    
    return df


class MRIDataset(Dataset):
    def __init__(self, root_img, labelPath , transform=None, Interpolation = False, pad=False,
                 Filter_Axial=True,Filter_type =False,balanced = False):
        
        self.root_img = root_img
        self.transform = transform
        self.Interpolation = Interpolation
        self.pad = pad
        self.label = Type_Axial_filter(root_img,labelPath)
        if Filter_Axial:
            self.label = self.label[self.label['Orientation']=='Axial']
        if Filter_type:
            self.label = self.label[self.label['Modality']== Filter_type]
            self.num_anbnormal = self.label[self.label['prediction']==1].shape[0]
        else:
            self.num_anbnormal = self.label[self.label['prediction']==1].shape[0]
            
        if balanced:
            self.label = pd.concat([self.label[self.label['prediction']==1],
                                    self.label[self.label['prediction']==0].iloc[:self.num_anbnormal]])
        
        self.image_paths = []
        self.labels = []

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = self.label.iloc[idx]['prediction']
        patient_path = os.path.join(self.root_img, self.label.iloc[idx]['SeriesInstanceUID'])
        
        if self.Interpolation:
            images = self.interpolate_slices(self.read_dicom(patient_path),20)
            images = np.array(images,dtype=float)
            images = images / images.max()
        elif self.pad:
            images = np.stack(self.read_dicom(patient_path),axis = 0,dtype=float)
            images = images / images.max()
            images = self.padding(images)
        else:
            images = np.stack(self.read_dicom(patient_path),axis = 0,dtype=float)
            images = images / images.max()
        
        
            
            
        images = torch.from_numpy(images)
        
        if self.transform:
            images = self.transform(images)
        
        return images, label
    
    def read_dicom(self,path):
        dicom_files = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path) if f.endswith('.dcm')]
        dicom_files_with_location = []
        
        for dicom_file in dicom_files:
            try:
                location = self.get_slice_location(dicom_file)
                dicom_files_with_location.append((location, dicom_file))
            except ValueError as e:
                print(e)
                
        dicom_files_with_location.sort(key=lambda x: x[0])
        sorted_dicom_files = [cv2.resize(file.pixel_array,(288,288)) for _, file in dicom_files_with_location]
    
        return sorted_dicom_files
                
    def get_slice_location(self,dicom_data):
        """
        Extracts the slice location from a DICOM file.
        """
        # Try to get the SliceLocation attribute first, fallback to ImagePositionPatient if unavailable
        try:
            return dicom_data.SliceLocation
        except AttributeError:
            try:
                return dicom_data.ImagePositionPatient[2]  # Assuming axial slices, z-coordinate
            except AttributeError:
                raise ValueError(f"Cannot determine slice location for file: {dicom_data}")
    
    def interpolate_slices(self,dicom_series, target_num_slices):
        image_data = np.stack(dicom_series, axis=0)
        original_num_slices = image_data.shape[0]
        zoom_factors = [target_num_slices / original_num_slices] + [1] * (image_data.ndim - 1)
        interpolated_data = zoom(image_data, zoom_factors, order=1)  # Linear interpolation
        return interpolated_data
    
    def padding(self,dicoms_file):
        m,n,k = dicoms_file.shape
        pad = np.zeros((20-m,n,k))
        return np.concatenate((dicoms_file,pad),axis=0)
    
    
        
imgs_path = r'D:\Project\Data\IAAA\data'
labelPath = r'D:\Project\Data\IAAA\train.csv'
    
MRI = MRIDataset(imgs_path,labelPath,Interpolation =False,
                     pad = True,
                     Filter_type='T2W_FLAIR',
                     ) 

print('number of Data: ',len(MRI))
print(MRI[0][0].shape)