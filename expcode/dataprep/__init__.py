
import json
from os.path import join
#assuming running from root
with open('./expcode/dataprep/data_prep_config_example.json', 'r') as f:
	config = json.load(f)

datasets_path = config["datasets"]
fec_config = config["FEC"]
proj_dir = fec_config["proj_dir"]
db_file = join(proj_dir,fec_config["db_file"])
rawData_dir = join(datasets_path,fec_config["rawData_dir"])
processedData_dir = join(proj_dir,fec_config["processedData_dir"])
csv_file_dict = {k : join(rawData_dir,v) for (k,v) in fec_config["csv_file_dict"].items()}
image_file_dict = {k : join(rawData_dir,v) for (k,v) in fec_config["image_file_dict"].items()}
