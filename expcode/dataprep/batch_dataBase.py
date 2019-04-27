save_dir = "./FEC_dataset"
process_dir = "./process_fec"
train_csv = "faceexp-comparison-data-train-public.csv"
test_csv = "faceexp-comparison-data-test-public.csv"
tt= ("train", "test")
lineskip=18
csvRange=(0,4000,lineskip)

import sqlite3
from sqlite3 import Error
import pandas as pd
from .batch_data_prep import ImageDataPrepFEC

def create_connection(db_file):
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

def create_table(conn, create_table_sql):
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

def pandas_csv_mani(csv_file):
  pass
def test():
  i = ImageDataPrepFEC()
  db_file = "test.db"
  c = create_connection(db_file)
