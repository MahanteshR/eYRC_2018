# import required libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data.dataset import Dataset
import sys
import csv


def create(dspath, dnpath):
    myFileList = createFileList(os.path.join("../", os.path.abspath(dspath)))
    csv_dst = "dataset_attr.csv"
    dnpath = os.path.join(dnpath, csv_dst)
    for file in myFileList:
        pathname = file.split(os.sep)
        img_file = Image.open(file)

        # get original image parameters...

        # Make image rgb
        img_rgb = img_file.convert('RGB')

        # Save rgb values
        value = np.asarray(img_rgb.getdata(), dtype=np.int).reshape(
            (3, img_rgb.size[1], img_rgb.size[0]))
        value = value.flatten()
        # Here the first column contains the labels
        value = np.append(pathname[len(pathname)-2], value)
        value = value.flatten()
        value = np.append(file, value)
        value.flatten()
        with open(dnpath, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(value)
    return True


def createFileList(myDir, format='.jpg'):
    fileList = []
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList


def create_meta_csv(dataset_path, destination_path):
    """Create a meta csv file given a dataset folder path of images.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The purpose behind creating this file is to allow loading of images on demand as required. Only those images 
    required are loaded randomly but on demand using their paths.
    
    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta file if None provided, it'll store file in dataset_path

    Returns:
        True (bool): Returns True if 'dataset_attr.csv' was created successfully else returns an exception
    """

    # Change dataset path accordingly
    # if os.path.exists(os.path.join(dataset_path, 'dataset_attr.csv')):
    #     os.remove(os.path.join(dataset_path, 'dataset_attr.csv'))
    if not os.path.exists(os.path.join(dataset_path, "./dataset_attr.csv")):
        # Make a csv with full file path and labels
        # change destination_path to DATASET_PATH if destination_path is None 
        if destination_path == None:
            destination_path = dataset_path

        # write out as dataset_attr.csv in destination_path directory
        if create(dataset_path, destination_path) == True:
            return True
        else:
            return False
    return True
        

def create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True, split=None):
    """Create a meta csv file given a dataset folder path of images and loads it as a pandas dataframe.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The function will return pandas dataframes for the csv and also train and test splits if you specify a 
    fraction in split parameter.
    
    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta csv file
        randomize (bool, optional): Randomize the csv records. Defaults to True
        split (double, optional): Percentage of train records. Defaults to None

    Returns:
        dframe (pandas.Dataframe): Returns a single Dataframe for csv if split is none, else returns more two Dataframes for train and test splits.
        train_set (pandas.Dataframe): Returns a Dataframe of length (split) * len(dframe)
        test_set (pandas.Dataframe): Returns a Dataframe of length (1 - split) * len(dframe)
    """
    if create_meta_csv(dataset_path, destination_path=destination_path):
        dframe=pd.read_csv(os.path.join(destination_path, 'dataset_attr.csv'))
    # shuffle if randomize is True or if split specified and randomize is not specified 
    # so default behavior is split
    if randomize == True or (split != None and randomize == None):
        # shuffle the dataframe here
        dframe.sample(frac=1)
        pass

    if split != None:
        train_set, test_set = train_test_split(dframe, test_size=(1. - split))
        return dframe, train_set, test_set 
    
    return dframe

# # def train_test_split(dframe, split_ratio):
# #     """Splits the dataframe into train and test subset dataframes.

# #     Args:
# #         split_ration (float): Divides dframe into two splits.

# #     Returns:
# #         train_data (pandas.Dataframe): Returns a Dataframe of length (split_ratio) * len(dframe)
# #         test_data (pandas.Dataframe): Returns a Dataframe of length (1 - split_ratio) * len(dframe)
# #     """
# #     # divide into train and test dataframes
# #     train_data, test_data = train_test_split(dframe, test_size=(1.-split_ratio))
# #     return train_data, test_data

class ImageDataset(Dataset):
    """Image Dataset that works with images
    
    This class inherits from torch.utils.data.Dataset and will be used inside torch.utils.data.DataLoader
    Args:
        data (str): Dataframe with path and label of images.
        transform (torchvision.transforms.Compose, optional): Transform to be applied on a sample. Defaults to None.
    
    Examples:
        >>> df, train_df, test_df = create_and_load_meta_csv_df(dataset_path, destination_path, randomize=randomize, split=0.99)
        >>> train_dataset = dataset.ImageDataset(train_df)
        >>> test_dataset = dataset.ImageDataset(test_df, transform=...)
    """

    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.transform = transforms.ToTensor()
        # Read the csv file
        self.data = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data.iloc[:, 0])
        # Second column is the labels
        self.classes = np.asarray(self.data.iloc[:, 1])

    def __getitem__(self, index):
        img_path=self.image_arr
        # Open image
        image = Image.open(img_path)

        # Get label(class) of the image based on the cropped pandas column
        label = self.classes[index]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    # test config
    dataset_path = './Data/fruits'
    dest = './Data/fruits'
    classes = 5
    total_rows = 4323
    randomize = True
    clear = True
    
    # test_create_meta_csv()
    df, trn_df, tst_df = create_and_load_meta_csv_df(dataset_path, destination_path=dest, randomize=randomize, split=0.99)
    # print(df.describe())
    # print(trn_df.describe())
    # print(tst_df.describe())

