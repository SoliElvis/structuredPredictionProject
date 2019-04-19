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


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str,
                    default="FEC_dataset/faceexp-comparison-data-train-public.csv",
                    help="Path to  the data file")
parser.add_argument("--save_path", type=str, default="FEC_dataset")
parser.add_argument("--dim", type=int, default=32, help="The desired dimension for the images")
args = parser.parse_args()


def load_data():
    # download the training set and test set
    ssl._create_default_https_context = ssl._create_unverified_context
    with open(args.data_path) as csv_file:
        csv_reader = np.genfromtxt(csv_file, delimiter=',')

        # make the save directory if not created already
        if not os.path.isdir(args.save_path):
            os.mkdir(args.save_path)
            os.mkdir(os.path.join(args.save_path, "train"))
            os.mkdir(os.path.join(args.save_path, "test"))

        for dataset in ["train", "test"]:
            for row in csv_reader:
                for i in range(3):
                    # download image
                    url = row[i * 5]
                    print(url)
                    file_name = os.path.basename(url)
                    file_path = os.path.join(args.save_path, dataset, file_name)
                    if not os.path.isfile(file_path):
                        try:
                            wget.download(url, file_path)
                        except urllib.error.HTTPError:
                            continue

                        # load the image and crop to only get the face box
                        im = Image.open(file_path)
                        w, h = im.size
                        corner = (np.rint(float(row[i * 5 + 1]) * w), np.rint(float(row[i * 5 + 3]) * h))
                        area = (corner[0], corner[1], np.rint(float(row[i * 5 + 2]) * w),
                                np.rint(float(row[i * 5 + 4]) * h))

                        try:
                            cropped = im.crop(area)
                            cropped.save(file_path)
                        except SystemError:
                            continue


def get_feat():
    # download the training set and test set
    ssl._create_default_https_context = ssl._create_unverified_context
    with open(args.data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        # make the save directory if not created already
        if not os.path.isdir(args.save_path):
            os.mkdir(args.save_path)
            os.mkdir(os.path.join(args.save_path, "train"))
            os.mkdir(os.path.join(args.save_path, "test"))

        for dataset in ["train", "test"]:
            with open(os.path.join("testing", dataset + ".npy"), "ab") as f:
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
                        crop = np.asarray(ImageOps.fit(crop, (args.dim, args.dim), Image.ANTIALIAS), dtype=np.float32)

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
                np.save(f, to_save)
                f.close()
        csv_file.close()


start = time.time()
get_feat()
print("Elapsed time", time.time() - start)
