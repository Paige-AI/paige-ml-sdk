python -m paige.ml_sdk --data.tune_dataset_path .run_dummy/dataset.csv --data.train_embeddings_dir .run_dummy/ --data.label_columns [cancer,precursor] --data.embeddings_filename_column slide --model BinClsAgata --data.train_dataset_path .run_dummy/dataset.csv --model.label_names [cancer,precursor] --model.in_features 5 --model.layer1_out_features 3 --model.layer2_out_features 3

