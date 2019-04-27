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
csv_file_dict = {"train" :
                  "/home/sole/project/expcode/dataprep/FEC_dataset/faceexp-comparison-data-train-public.csv"}
from sqlalchemy import create_engine

# sql_engine = create_engine('sqlite:///test.db', echo=False)
# connection = sql_engine.raw_connection()
# working_df.to_sql('data', connection,index=False, if_exists='append')

#TODO: factor out formater
# Idea: a bunch of csv files in a dict and one db file path

class csv_to_sql():
  def __init__(self,csv_file_dict : Dict[str,str],db_file_path : str):
    self.csv_file_dict = csv_file_dict
    self.db_file_path = db_file_path
    self.df = None
    self.db= None
    try:
      self.db = self._create_connection(db_file_path)
    except Exception as e:
      print(e)


  #csv stuff
  def _pandas_load_csv(self,csv_file_key):
    try:
      self.df = pd.read_csv(csv_file_dict[csv_file_key],sep=',',header=None,error_bad_lines=False)
    except KeyError as kerr:
      print(kerr)
      self.df = None
    return self.df

  def _format_panda(self,csv_file_key,formater=None):

    df = self._pandas_load_csv(csv_file_key)
    _cols  = [[str(i) + "_url",
                str(i) + "_faceCrop_1", str(i) + "_faceCrop_2",
                str(i) + "_faceCrop_3", str(i) + "_faceCrop_4"] for i in range(0,11,5)]
    _cols = [item for sublist in _cols for item in sublist]
    _cols.append("_".join(["16","triple_config"]))
    for i in range(17,23):
      _cols.append("_".join([str(i),"Anotator_id"]))
      _cols.append("_".join([str(i),"note"]))

    df.columns = _cols
    df.name = "_".join([c[0] for c in _cols])
    self.df = df
    return self.df

  #db stuff
  def _create_connection(self,db_file_path=None):
    db_file_path = db_file_path or self.db_file_path
    try:
      self.db= sqlite3.connect(db_file_path)
      return self.db
    except Error as e:
      print(e)
    return None


  #df needs a name which will be the one of the table
  def export_to_sql(self,df=None,db=None):
    df = df or self.df
    db = db or self.db
    if not connection_test(db):
      print("db closed"); pass

    if not dataframe_test(df):
      print("dataframe fucked"); pass

    df.to_sql(df.name,db)
    return df,db


def connection_test(db):
  return db is not None
def dataframe_test(df):
  return df is not None





def test():
  db_file_path = os.path.join("./","test.db")
  plug = csv_to_sql(csv_file_dict,db_file_path)

  return plug












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
