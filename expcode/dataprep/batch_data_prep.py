import numpy as np
import numba as nb
import csv
import os
import argparse
import wget
import ssl
import PIL
from PIL import Image
from PIL import ImageOps
import urllib
import timeit
from io import BytesIO
import requests
import wget
import time

from typing import List
from IPython.core import debugger
debug = debugger.Pdb().set_trace
import pathlib
from os import listdir
from os.path import isfile
from itertools import islice
from collections import namedtuple

save_dir = "./FEC_dataset"
process_dir = "./process_fec"
train_csv = "faceexp-comparison-data-train-public.csv"
test_csv = "faceexp-comparison-data-test-public.csv"
tt= ("train", "test")
lineskip=18
csvRange=(0,4000,lineskip)

#utilities
def tt_join_paths(prefixes):
  r = [os.path.join(p,t) for p in prefixes for t in tt]
  return r
def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--save_dir", type=str, default="./FEC_dataset")
  parser.add_argument("--data_path", type=str,
                      default="./FEC_dataset/faceexp-comparison-data-train-public.csv",
                      help="Path to  the data file")
  parser.add_argument("--dim", type=int, default=32,
                      help="The desired dimension for the images")
  parser.add_argument("--images_path", type=str,
                      default="./face_images",
                      help="Path to directory to store images")
  args, unkown = parser.parse_known_args()
  return args

def createFolder(directory_list : List[str]):
  print(directory_list)
  for directory in directory_list:
    try:
      if not os.path.exists(directory):
        os.makedirs(directory)
    except OSError:
      print ('Error: Creating directory. ' +  directory)

  return directory_list

#vrmt batard
def seeker(it,spamreader,skip):
  for id, row in enumerate(spamreader):
    next(it)
    if id == skip:
      break
  return it

class ImageDataPrep:
  def __init__(self,save_dir,process_dir,dim=32):
    self.save_dir = save_dir
    self.imagesDir = os.path.join(process_dir, "images")
    self.dataDir = os.path.join(process_dir, "data")

    self.csvDataSets = [os.path.join(self.process_dir, f) for f in [train_csv,test_csv]]
    self.csvDataPathDict = {"train" : self.csvDataSets[0],
                         "test"  : self.csvDataSets[1]}
    self._prep_file_system()
    self.dim = dim

  def _prep_file_system(self):
    dirsToCreate = tt_join_paths([self.imagesDir,self.dataDir])
    createFolder(dirsToCreate)
    return dirsToCreate


class ImageDataPrepFEC(ImageDataPrep):
  def __init__(self,save_dir="./FEC_dataset",process_dir="./process_fec/"):
    self.save_dir = save_dir
    self.process_dir = process_dir
    self.local_state_dict = None
    self.urlSlots = [0,5,10]
    ImageDataPrep.__init__(self,self.save_dir,self.process_dir)

  def batch_download_images(self,spamreader=True,skip=4000,stopLine=5000):
    urlSlots = [0,5,10]
    ssl._create_default_https_context = ssl._create_unverified_context
    for testOrTrainStr, path in self.csvDataPathDict.items():
      with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        it = enumerate(csv_reader)

        if (spamreader):
          spamreader = csv.reader(csv_file, delimiter=',')
          for id, row in enumerate(spamreader):
            next(it)
            if id ==  skip:
              break

        for id, row in it:
          print(id)
          if id >= stopLine:
            break
          for slot in urlSlots:
            url = row[slot]
            response = requests.get(url)

            if (response.status_code != 200):
              print(" not 200")
              break

            path = os.path.join(self.imagesDir,testOrTrainStr, str(id) + "-" + str(slot) + ".jpg")
            if not pathlib.Path(path).is_file():
              try:
                wget.download(url, out=path)
              except :
                print(path)
            print(path)

  def process_data(self,skip=4000,stopLine=5000,to_save=None):
    to_save = np.empty([0, (3 * self.dim) ** 2 + 7])
    self.urlSlots = [0,5,10]
    tt = ("train", "test")
    for testOrTrainStr in tt:
      dataPath = self.csvDataPathDict[testOrTrainStr]
      p = os.path.join(self.dataDir,testOrTrainStr,"image_processed.npy")
      try:
        with open(dataPath) as csv_file:
          f = open(p, "w+")
          csv_reader = csv.reader(csv_file, delimiter=',')
          it = enumerate(csv_reader)
          spamreader = csv.reader(csv_file, delimiter=',')
          it = seeker(it,spamreader,skip)
          for id, row in it:
            print(id)
            if (id >= stopLine):
              break
            try:
              self._process_image_triple(id,row,it,to_save)
            except ValueError as e:
              print(e)

      except FileExistsError:
        print("exists")
      except Exception as e:
        print(e)
      finally:
        print("skipppp")
        f.close()


  def _process_image_triple(testOrTrainStr,id,row,it,to_save): # retunns the cropped array tuple
    image_path_triple = []
    processed_asarray_triple = []
    for slot in self.urlSlots:
      imagePath = os.path.join(self.imagesDir,testOrTrainStr, str(id) + "-" + str(slot) + ".jpg")
      if not os.path.isfile(image_path):
        raise ValueError
      image_path_triple.append(imagePath)

    for i in self.urlSlots:
      im = Image.open(BytesIO(image_path_triple[0]))
      w, h = im.size
      left, up = np.rint(float(row[i + 1]) * w), np.rint(float(row[i + 3]) * h)
      right, down = np.rint(float(row[i + 2]) * w), np.rint(float(row[i + 4]) * h)
      area = (int(left), int(up), int(right), int(down))
      crop = im.crop(area)
      # resize the image to the right proportion
      crop = np.asarray(ImageOps.fit(crop, (self.dim, self.dim), Image.ANTIALIAS), dtype=np.float32)
      processed_asarray_triple.append(crop)

    if len(processed_asarray_triple) == 3:
      raise ValueError

    # extract features and label
    features = np.stack(im_tup).ravel()
    label = np.asarray([row[17 + i * 2] for i in range(6)], dtype=np.float32)

    if row[15] == "ONE_CLASS_TRIPLET":
        label = np.concatenate(([1.], label)).astype("float32")
    elif row[15] == "TWO_CLASS_TRIPLET":
        label = np.concatenate(([2.], label)).astype("float32")
    else:
        label = np.concatenate(([3], label)).astype("float32")

    im_tup = np.concatenate((features, label)).reshape([1, -1])

    # add the row to_save
    to_save = np.concatenate((to_save, im_tup))

  def _check_local_images(self):
    present_images = {"train" : list(), "test" :list()}
    for testOrTrainStr in present_images:
      mypath = os.path.join(self.imagesDir,testOrTrainStr)
      for f in listdir(mypath):
        imageCodeFromFileName, extension = os.path.splitext(f)
        present_images[testOrTrainStr].append(imageCodeFromFileName)
    return present_images



    # extract features and label
    try:
      features = np.stack(im_tup).ravel()
    except ValueError:
      raise ValueError

    label = np.asarray([row[17 + i * 2] for i in range(6)], dtype=np.float32)
    if row[15] == "ONE_CLASS_TRIPLET":
      label = np.concatenate(([1.], label)).astype("float32")
    elif row[15] == "TWO_CLASS_TRIPLET":
      label = np.concatenate(([2.], label)).astype("float32")
    else:
      label = np.concatenate(([3], label)).astype("float32")

    im_tup = np.concatenate((features, label)).reshape([1, -1])
    # add the row to_save
    to_save = np.concatenate((to_save, im_tup))
    return to_save

  def _troubleshoot(self,spamreader=False):
    csv_file = open(self.csvDataSets[0])
    if spamreader:
      spamreader = csv.reader(csv_file, delimiter=',')
      for row in spamreader:
        print(', '.join(row))

    csv_reader = csv.reader(csv_file, delimiter=',')
    return csv_reader


def main():
  start = time.time()
  test = ImageDataPrepFEC()
  local_images=None
  # local_images = test._check_local_images()
  # test.batch_download_images()
  test.process_data()
  return test, local_images
  print("Elapsed time", time.time() - start)

if __name__ == "__main__":
  main()
