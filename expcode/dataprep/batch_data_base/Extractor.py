
import expcode.dataprep.batch_data_prep as prep

import os
import sqlite3
import pandas as pd
from sqlite3 import Error
from typing import List, Dict
import itertools
import pipe
# from sqlalchemy import create_engine
# sql_engine = create_engine('sqlite:///test.db', echo=False)
# connection = sql_engine.raw_connection()
# working_df.to_sql('data', connection,index=False, if_exists='append')
#Utilities
def face_crop_df_formater(df):
  _cols  = [[str(i)  + "_url",
              str(i) + "_faceCrop_1", str(i) + "_faceCrop_2",
              str(i) + "_faceCrop_3", str(i) + "_faceCrop_4"] for i in range(0,11,5)]
  _cols = [item for sublist in _cols for item in sublist]
  _cols.append("_".join(["16","triple_config"]))
  for i in range(17,23):
    _cols.append("_".join([str(i),"Anotator_id"]))
    _cols.append("_".join([str(i),"note"]))

  df.columns = _cols
  df.name = "_".join([c[0] for c in _cols])
  return df

def connection_test(db):
  return db is not None
def dataframe_test(df):
  return df is not None

#
# csv -> dataframe -> db -> -> ETL of tables -> .npy
#                     |      \
#                    sync      -> dataframe -> .npy
#                     db!!!
#

# Idea: a bunch of csv files in a dict and one db file path
# TODO : what to do if db exists, error exception in general,
# TODO : sanitize closing db
# db_file_path should be system wide

class Extractor_csv_to_sql():
  def __init__(self,csv_file_dict : Dict[str,str], db_file_path : str):
    self.csv_file_dict = csv_file_dict
    self.db_file_path = db_file_path
    self.df = None
    self.db= None
    try:
      self.db = self._create_connection()
    except Exception as e:
      print(e)

  def _create_connection(self):
    try:
      self.db= sqlite3.connect(self.db_file_path)
      return self.db
    except Error as e:
      print(self.db_file_path)
      print(e)
    return None

  def export_to_sql(self,db=None):
    df_dict = {}
    db = db or self.db
    if not connection_test(db):
      print("db closed"); return None

    for name, path in self.csv_file_dict.items():
      df_dict[name] = self._pandas_load_csv(name) | self._format_panda(name)
      df_dict[name].name = name
      if not dataframe_test(df_dict[name]):
        print("dataframe fucked"); continue
      df_dict[name].to_sql(name,db)

    return df_dict,db

  def _pandas_load_csv(self,csv_file_key):
    try:
      self.df = pd.read_csv(self.csv_file_dict[csv_file_key],sep=',',error_bad_lines=False)
    except KeyError as kerr:
      print(kerr)
      self.df = None
    return self.df

  def _format_panda(self,csv_file_key,formater=face_crop_df_formater):
    df = self._pandas_load_csv(csv_file_key)
    if formater is not None:
      self.df = formater(df)
    return self.df


#Needs to run from root of project
def extract_csv_fec():
  #move all that to json file
  proj_dir = "./expcode"
  db_file_path = os.path.join(proj_dir,"fec.db")
  save_dir = os.path.join(proj_dir,"dataprep/FEC_dataset")
  process_dir = os.path.join(proj_dir,"process_dev")

  csv_file_dict = {"train-fec": os.path.join(save_dir, "faceexp-comparison-data-train-public.csv"),
                   "test-fec" : os.path.join(save_dir, "faceexp-comparison-data-test-public")}

  image_file_dict = {"train-fec" : os.path.join(process_dir,"images/train"),
                     "test-fec"  : os.path.join(process_dir, "images/test")}

  plug = Extractor_csv_to_sql(csv_file_dict,db_file_path)
  plug.export_to_sql()



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
