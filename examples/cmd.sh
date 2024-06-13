python -m paige.ml_sdk \
fit \
--data.train_dataset_path dataset.csv \
--data.tune_dataset_path dataset.csv \
--data.embeddings_dir . \
--data.label_columns [cancer,precursor] \
--data.embeddings_filename_column slide \
--model BinClsAgata \
--model.in_features 5 \
--model.layer1_out_features 3 \
--model.layer2_out_features 3

