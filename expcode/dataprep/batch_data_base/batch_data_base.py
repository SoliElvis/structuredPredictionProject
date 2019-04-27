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

csv_file_dict = {"train" :
                 "/home/sole/project/expcode/dataprep/FEC_dataset/faceexp-comparison-data-train-public.csv"}

db_file_path = "./test.db"
class csv_to_sql():

  def __init__(self,csv_file_dict : Dict[str,str],db_file_path : str):
    self.csv_file_dict = csv_file_dict
    self.db_file_path = self.db_file_path

  def _create_connection(db_file):
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

  def _create_table(conn, create_table_sql):
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

  def _pandas_load_csv(csv_file):
    df = pandas.csv_read(csv_file)

def test():
  db_file_path = os.path.join("./","test.db")
  c = create_connection(db_file_path)
  plug = csv_to_sql(csv_file_dict,db_file_path)
  plug_df = plug._pandas_load_csv(csv_file_dict["train"])
