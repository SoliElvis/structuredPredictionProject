proj_dir = "./expcode"
db_file_path = os.path.join(proj_dir,"fec.db")
save_dir = os.path.join(proj_dir,"dataprep/FEC_dataset")
process_dir = os.path.join(proj_dir,"process_dev")
csv_file_dict = {"train_fec": os.path.join(save_dir, "faceexp-comparison-data-train-public.csv"),
									"test_fec" : os.path.join(save_dir, "faceexp-comparison-data-test-public.csv")}
image_file_dict = {"train_fec" : os.path.join(process_dir,"images/train"),
										"test_fec"  : os.path.join(process_dir, "images/test")}
