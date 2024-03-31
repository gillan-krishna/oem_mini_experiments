import os
from pathlib import Path
import numpy as np
from skimage.io import imread
from tqdm import tqdm
import csv
from glob import glob

class_grey_oem = {
    0:"Unknown",
    1:"Bareland",
    2:"Rangeland",
    3:"Developed space",
    4:"Road",
    5:"Tree",
    6:"Water",
    7:"Agriculture",
    8:"Building",
}

def count_unique_in_list(fn_list):
    total_pixels = 0
    unique_counts = {}
    
    # Iterate over each file in the directory
    # for filepath in tqdm(fn_list):
    for filepath in fn_list:
        # Read the TIFF image
        image = imread(filepath)
        total_pixels += image.size
        # Flatten the image into a 1D array
        flattened_image = image.flatten()
        # Count unique values and update unique_counts
        unique_values, counts = np.unique(flattened_image, return_counts=True)
        for value, count in zip(unique_values, counts):
            if value in unique_counts:
                unique_counts[value] += count
            else:
                unique_counts[value] = count
    unique_percentages = {value: (count / total_pixels) * 100 for value, count in unique_counts.items()}

    return unique_percentages

def export_to_csv(data, filename):
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class #', 'Class', 'Percentage'])
        for class_name, percentage in data.items():
            writer.writerow([class_name, class_grey_oem[class_name], f"{percentage:.2f}%"])

if __name__ == '__main__':
    # OEM_DATA_DIR = Path('/home/gillan/mini-oem/data/processing/OpenEarthMap_Mini')
    # TRAIN_LIST = OEM_DATA_DIR.joinpath('train.txt')
    # VAL_LIST = OEM_DATA_DIR.joinpath('val.txt')

    # fns = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/labels/" in str(f)]
    # train_fns = [str(f) for f in fns if f.name in np.loadtxt(TRAIN_LIST, dtype=str)]
    # val_fns = [str(f) for f in fns if f.name in np.loadtxt(VAL_LIST, dtype=str)]
    # train_fns = glob('/home/gillan/mini-oem/data/balanced/train/labels/*.tif')
    val_fns = glob('/home/gillan/mini-oem/data/balanced/train/labels/*.tif')


    # unique_percentages = count_unique_in_list(train_fns)
    # output_csv_file = Path('/home/gillan/mini-oem/reports/split_train.csv')
    # export_to_csv(unique_percentages, output_csv_file)

    unique_percentages = count_unique_in_list(val_fns)
    output_csv_file = Path('/home/gillan/mini-oem/reports/split_train.csv')
    export_to_csv(unique_percentages, output_csv_file)