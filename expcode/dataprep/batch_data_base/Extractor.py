import expcode.dataprep.batch_data_prep as prep
from expcode.dataprep import dp_fec, dp_pkl,dp_trl, dataParams

import os
import sqlite3
from sqlite3 import Error
import pandas as pd
from typing import List, Dict
import itertools
import pipe
import pickle
# import cPickle
import signal
from contextlib import contextmanager

import asyncio
import mmap
import functools
import nltk

#Globals-----------------------------------------------------

testOrTrain = ("train","test")

#Utilities---------------------------------------------------

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
  return df

def connection_test(db):
  return db is not None
def dataframe_test(df):
  return df is not None

class TimeoutException(Exception): pass
@contextmanager
def time_limit(seconds):
  def signal_handler(signum, frame):
    raise TimeoutException("Timed out!")
  signal.signal(signal.SIGALRM, signal_handler)
  signal.alarm(seconds)
  try:
    yield
  finally:
    signal.alarm(0)


#Extractors --------------------------------------------------

class ExtractorBase():
  def __init__(self,db_file_path : str):
    self.db_file_path = db_file_path
    self.db = None
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


class Extractor_csv_to_lite(ExtractorBase):
  def __init__(self,db_file_path : str, csv_file_dict : Dict[str,str]):
    ExtractorBase.__init__(db_file_path)
    self.csv_file_dict = csv_file_dict
    self.df_dict = None

  def _pandas_load_csv(self,csv_file_key):
    try:
      df = pd.read_csv(self.csv_file_dict[csv_file_key],sep=',',error_bad_lines=False)
      df = self._format_panda(df)
      df.name = csv_file_key
      return df
    except KeyError as kerr:
      print(kerr)
    return df

  def _format_panda(self,df,formater=face_crop_df_formater):
    if formater is not None:
      df = formater(df)
    return df

  def first_export_to_lite(self,db=None):
    df_dict = {}
    db = db or self.db
    if not connection_test(db):
      print("db closed"); return None

    for name, path in self.csv_file_dict.items():
      df_dict[name] = self._pandas_load_csv(name)
      if not dataframe_test(df_dict[name]):
        print("dataframe fucked"); continue
      df_dict[name].to_sql(name,db,if_exists='append')

    return df_dict,db


class Extractor_pickle_to_lite(ExtractorBase):
  def __init__(self,dp_pickle):
    ExtractorBase.__init__(dp_pickle.db_file)
    self.pickdic = dp_pickle.pickdic
    self.df_dict = None
    self.content_inMem = {lang[0]:list(), lang[1]:list()}
  def _load_pickle_file(self):
    for l in lang:
      with open(self.pickdic[l], 'rb') as f:
        unpick = pickle.Unpickler(f)
        temp_content = unpick.load()
        yield temp_content
        self.content_inMem[l].append(temp_content)

    return self.content_inMem



#sql table : 20 phrases, varchar
lang = ("fr","en")
class Extractor_textFile_to_lite(ExtractorBase):
  def __init__(self,dp_trl):
    ExtractorBase.__init__(self,dp_trl.db_file)
    self.dp_trl = dp_trl
    self.text_fileDict = dp_trl.text_fileDict
    self.corpus= []
    try:
       self.db.execute("create table fr_batch(id,sent)")
       self.db.execute("create table en_batch(id,sent)")
       self.db.execute("create table fr(id,sent)")
       self.db.execute("create table en(id,sent)")
    except Exception as e:
      print(e)

  def mass_inserter(self):
    #open DB in append mode
    for l in lang:
      with open(self.text_fileDict[l], 'r') as f:
        chunks = {}
        for id,chunk in zip(range(10000),iter(functools.partial(f.read,1024), b'')):
          tableName = l + "_batch" + "(id,sent)"
          sents = ''.join([c for c in chunk])
          sents = '|'.join(nltk.sent_tokenize(sents))
          payload = (id,sents)
          self.db.execute("insert into " + tableName  + " values (?,?)",payload)
    return self.db

  def etl_1(self):
    for l in lang:
      source = l + "_batch"
      target = l + "_1"
      counter = 0
      text = self.db.execute("select * from " + source).fetchone()
      for row in text:
        print(row)
        print('\n')







def main():
  ex = Extractor_textFile_to_lite(dp_trl)
  return ex

def test():
  time_limit_sec = 60
  try:
    with time_limit(time_limit_sec):
      pk_test = Extractor_pickle_to_lite(dp_pkl)
      for p in pk_test._load_pickle_file():
        print("----------------------------------------------------------------")
        print("----------------------------------------------------------------")
        print(p)
  except TimeoutException as e:
    print("Timed out!")
  [print(t) for t in pk_test._load_pickle_file()]
  return pk_test
  return dp_fec,dp_pkl

def test2():
  return Extractor_textFile_to_lite(dp_trl)
#Needs to run from root of project
def extract_csv_fec():
  plug = Extractor_csv_to_lite(dp_fec.csv_file_dict,dp_fec.db_file)
  plug.first_export_to_lite()
  return plug












####################################################
# class Blob_manager_fs(db,db_to_file_format):
#   pass
# class Blob_manager_postGre(db,db_to_file_format):
#   pass
# class ETL_manager(db,aggregate_pattern):
#   pass
# class Numerical_transformation(db,numerical_pattern):
#   pass
#same process different tables
# class PostProcessor_csv_to_sql():
#   def __init__(self,db : str,table_namesz : List[str], update_postfix: str):
#     self.db = self._create_connection(db)
#     self.table_names = table_names
#     self.new_table_names = [os.path.join(n, update_postfix) for n in table_namesz]
#   def _create_connection(self):
#     try:
#       self.db= sqlite3.connect(self.db_file_path)
#       return self.db
#     except Error as e:
#       print(self.db_file_path)
#       print(e)
#     return None


# class Lite_to_postgres():
#   pass
