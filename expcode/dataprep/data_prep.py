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
import time

from typing import List

save_dir = "FEC_dataset"
train_csv = "faceexp-comparison-data-train-public.csv"
test_csv = "faceexp-comparison-data-test-public.csv"
tt= ("train", "test")

class ImageDataPrep:
	def __init__(self,save_dir):
		self.save_dir = save_dir
		self.imagesDir = os.path.join(save_dir, "images")
		self.dataDir = os.path.join(save_dir, "data")

		self.dataSets = [os.path.join(self.save_dir, f) for f in [train_csv,test_csv]]
		self.dataPathDict = {"train" : self.dataSets[0],
													"test"  : self.dataSets[1]}


	def prep_file_system(self):
		dirsToCreate = tt_join_paths([self.imagesDir,self.dataDir])
		createFolder(dirsToCreate)
		return dirsToCreate


class ImageDataPrepFEC(ImageDataPrep):
	def batch_download_images(self):
		urlSlots = [0,5,10]
		ssl._create_default_https_context = ssl._create_unverified_context
		for testOrTrain, path in dataPathDict:
			with open(path) as csv_file:
				csv_reader = csv.reader(csv_file, delimiter=',')
				for id, row in enumerate(csv_reader):
					for slot in urlSlots:
						response = requests.get(row[slot])
						if (reponse.status_code == 200): break
						if testOrTrain == "train":
							with open(os.path.join(dataPathDict["train"], "id" + "-" + "slot"), 'wb') as im:
								im.write(reponse.content)
						else :
							with open(os.path.join(dataPathDict["test"], "id" + "-" + "slot"), 'wb') as im:
								im.write(reponse.content)



 def download_data():
   # download the training set and test set
	ssl._create_default_https_context = ssl._create_unverified_context

	with open(args.data_path) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')

		for dataset in ["train", "test"]:
			with open(os.path.join(args.save_dir, dataset + ".npy"), 'ab') as f:
				to_save = np.empty([0, (3 * args.dim) ** 2 + 7])
				for id, row in enumerate(csv_reader):
					im_tup = []
					for i in [0, 5, 10]:
					# load the image
						response = requests.get(row[i])
						try:
							im = Image.open(BytesIO(response.content))
						except OSError:
							break

					# crop it to only keep face info
					w, h = im.size
					left, up = np.rint(float(row[i + 1]) * w), np.rint(float(row[i + 3]) * h)
					right, down = np.rint(float(row[i + 2]) * w), np.rint(float(row[i + 4]) * h)
					area = (int(left), int(up), int(right), int(down))
					crop = im.crop(area)

					# resize the image to the right proportion
					Crop = np.asarray(ImageOps.fit(crop,
																						(args.dim, args.dim), Image.ANTIALIAS), dtype=np.float32)

					# append the image to image tuple
					im_tup.append(crop)

					# if the image tuple does  not contain three images skip
					if not len(im_tup) == 3:
							continue

					# extract features and label
					try:
						features = np.stack(im_tup).ravel()
					except ValueError:
						continue

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

					if id > 10000 and dataset == "train":
							break
					elif id > 3000 and dataset == "test":
							break
						np.save(f.name, to_save)
						f.close()
						csv_file.close()


def main():
  prep_file_system(save_dir)
  start = time.time()
  download_data()
  print("Elapsed time", time.time() - start)

if __name__ == "__main__":
  main(args)

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

