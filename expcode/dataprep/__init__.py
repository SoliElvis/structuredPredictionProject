import json
from os.path import join
from dataclasses import dataclass


#assuming running from root
with open('./expcode/dataprep/data_prep_config_example.json', 'r') as f:
  config = json.load(f)

proj_dir = config["proj_dir"]
datasets_path = config["datasets"]

@dataclass
class dataP:
  db_file : str
  rawData_dir : str
  processedData_dir: str

class dataParams(dataP):
  def __init__(self,config_str):
    self.config = config[config_str]
    self.db_file = join(proj_dir,self.config["db_file"])
    self.rawData_dir = join(datasets_path,self.config["rawData_dir"])
    self.processedData_dir = join(proj_dir,self.config["processedData_dir"])

class dataParams_fec(dataParams):
  def __init__(self,config_str):
    dataParams.__init__(self,config_str)
    self.csv_file_dict = {k : join(self.rawData_dir,v) for (k,v) in self.config["csv_file_dict"].items()}
    self.image_file_dict = {k : join(self.rawData_dir,v) for (k,v) in self.config["image_file_dict"].items()}

class dataParams_pkl(dataParams):
  def __init__(self,config_str):
    dataParams.__init__(self,config_str)
    self.pickdic = {k : join(self.rawData_dir,v) for (k,v) in self.config["pickdic"].items()}


dp_fec = dataParams_fec("FEC")
dp_pkl = dataParams_pkl("PICKLE")
