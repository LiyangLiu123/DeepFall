python3 delete_missing_samples.py

python3 create_cs_data.py
python3 create_cv_data.py

python3 gen_ndarray_60cls.py ./cs_train
python3 gen_ndarray_60cls.py ./cs_test
python3 gen_ndarray_60cls.py ./cv_train
python3 gen_ndarray_60cls.py ./cv_test

python3 gen_ndarray_2cls.py ./cs_train
python3 gen_ndarray_2cls.py ./cs_test
python3 gen_ndarray_2cls.py ./cv_train
python3 gen_ndarray_2cls.py ./cv_test

python3 train_indrnn.py cs 60
python3 train_indrnn.py cs 2
python3 train_indrnn.py cv 60
python3 train_indrnn.py cv 2
