save_dir = "./FEC_dataset"
process_dir = "./process_fec"
train_csv = "faceexp-comparison-data-train-public.csv"
test_csv = "faceexp-comparison-data-test-public.csv"
tt= ("train", "test")
lineskip=18
csvRange=(0,4000,lineskip)

import os
import sqlite3
from sqlite3 import Error
import pandas as pd
import expcode.dataprep.batch_data_prep as prep
from typing import List, Dict
import itertools


db_file_path = "./test.db"

class csv_to_sql():
  def __init__(self,csv_file_dict : Dict[str,str],db_file_path : str):
    self.csv_file_dict = csv_file_dict
    self.db_file_path = db_file_path
    self.df = None

  def _create_connection(self,db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
      conn = sqlite3.connect(db_file)
      return conn
    except Error as e:
      print(e)
    return None

  def _create_table(self,conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
      c = conn.cursor()
      c.execute(create_table_sql)
    except Error as e:
      print(e)

  def _pandas_load_csv(self,csv_file_key):
    self.df = pd.read_csv(csv_file_dict[csv_file_key],sep=',',header=None,error_bad_lines=False)
    return self.df

  def _format_panda(self,formater,csv_file_key):
    self.df = self._pandas_load_csv(self,csv_file_key)
    __cols={0:"url", 1:"faceCrop_1", 2:"faceCrop_2", 3:"faceCrop_3", 4:"faceCrop_4"}
    for i in range(1,15,5):
      for __c in __cols.items():
        _cols[i+c[0]] = __c[1]

    _cols[16] = "triple_config"
    for i in range(17,28,2):
      _cols[i] = "Anotator_id"
      _cols[i+1] = "note"
    cols = {k-1: v for (k,v) in _cols.items() if k>0}
    plug_df.rename(index=int, columns=cols)


def test():
  csv_file_dict = {"train" :
                    "/home/sole/project/expcode/dataprep/FEC_dataset/faceexp-comparison-data-train-public.csv"}
  db_file_path = os.path.join("./","test.db")

  plug = csv_to_sql(csv_file_dict,db_file_path)
  c = plug._create_connection(db_file_path)

  plug._form
  return plug_df












# Each line in the CSV files has the following entries:
# ● URL of image1 (string)
# ● Top-left column of the face bounding box in image1 normalized by width (float)
# ● Bottom-right column of the face bounding box in image1 normalized by width (float)
# ● Top-left row of the face bounding box in image1 normalized by height (float)
# ● Bottom-right row of the face bounding box in image1 normalized by height (float)
# ● URL of image2 (string)
# ● Top-left column of the face bounding box in image2 normalized by width (float)
# ● Bottom-right column of the face bounding box in image2 normalized by width (float)
# ● Top-left row of the face bounding box in image2 normalized by height (float)
# ● Bottom-right row of the face bounding box in image2 normalized by height (float)
# ● URL of image3 (string)
# ● Top-left column of the face bounding box in image3 normalized by width (float)
# ● Bottom-right column of the face bounding box in image3 normalized by width (float)
# ● Top-left row of the face bounding box in image3 normalized by height (float)
# ● Bottom-right row of the face bounding box in image3 normalized by height (float)
# ● Triplet_type (string) - A string indicating the variation of expressions in the triplet.
# ● Annotator1_id (string) - This is just a string of random numbers that can be used to
# search for all the samples in the dataset annotated by a particular annotator.
# ● Annotation1 (integer)
# ● Annotator2_id (string)
# ● Annotation2 (integer)
