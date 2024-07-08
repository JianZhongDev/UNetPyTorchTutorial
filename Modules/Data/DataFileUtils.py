"""
FILENAME: DataFileUtils.py
DESCRIPTION: helper functions for data file handling
@author: Jian Zhong
"""

import os
import requests
import zipfile


## download from url
def download_url_to_local_dir(src_url, dst_dir_path = ".", chunk_size = 128, overwrite = False):
    # create local directory if not exists
    if not os.path.isdir(dst_dir_path):
        os.makedirs(dst_dir_path)

    # create destination file path
    dst_file_name = os.path.split(src_url)[-1]
    dst_file_path = os.path.join(dst_dir_path, dst_file_name)

    # check if file alreay exists
    if os.path.exists(dst_file_path) and not overwrite:
        return dst_file_path

    # dowload data from src_url to local using request module
    req = requests.get(src_url, stream = True)
    with open(dst_file_path, "wb") as dst_file:
        for chunk in req.iter_content(chunk_size = chunk_size):
            dst_file.write(chunk)

    return dst_file_path


## unzip file
def unzip_file_to_dir(src_zip_file_path, dst_dir_path = None, overwrite = False):
    
    # destination sub directory for the unzipped file 
    if dst_dir_path is None:
        dst_dir_path = os.path.split(src_zip_file_path)[0]
    dst_subdir_name = os.path.splitext(os.path.split(src_zip_file_path)[-1])[0]
    dst_subdir_path = os.path.join(dst_dir_path, dst_subdir_name)

    # create destination directory if not exisits 
    if not overwrite and os.path.isdir(dst_subdir_path):
        return dst_subdir_path
    
    if not os.path.isdir(dst_subdir_path):
        os.makedirs(dst_subdir_path)

    # unzip file
    with zipfile.ZipFile(src_zip_file_path, "r") as zip_file:
        zip_file.extractall(dst_subdir_path)

    return dst_subdir_path
    
    
    


