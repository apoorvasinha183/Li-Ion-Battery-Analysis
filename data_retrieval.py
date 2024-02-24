from urllib.request import urlretrieve
import progressbar
from zipfile import ZipFile 
import os
import shutil
# Credits : https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve
pbar = None

files = []
def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None
print("Downloading 5 battery Data")
#Battery Data Item 5 Nasa
url =("https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip")
fileName = "FiveBatteryData.zip"
files.append(fileName)
urlretrieve(url, fileName,show_progress)
print("Downloading charge/discharge battery Data-- WARNING : 1.1 GiB")
#Battery Data Item 11 NASA
fileName = "ChargeDischargeData.zip"
files.append(fileName)
url =(" https://phm-datasets.s3.amazonaws.com/NASA/11.+Randomized+Battery+Usage+Data+Set.zip")
urlretrieve(url, fileName,show_progress)
print("All data files downloaded! Unzipping the files in the data folder.")
for file in files:
    with ZipFile(file, 'r') as zObject: 
    
        # Extracting specific file in the zip 
        # into a specific location. 
        zObject.extractall(path="data//") 
    zObject.close() 
folder="data/"
subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
print("subfolders is ",subfolders)
print("Second round of unzips in progress!")
for subfolder in subfolders:
    fileList = [ f.path for f in os.scandir(subfolder) if f.is_file() ]
    for file in fileList:
        with ZipFile(file, 'r') as zObject: 
        
            # Extracting specific file in the zip 
            # into a specific location. 
            zObject.extractall(path="data//") 
        zObject.close() 
print("Complete!")        