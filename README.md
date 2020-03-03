# Memeory network for question answering in SQuAD 
CSE244 project

### Staring script:
run.ipynb

### Report:

### Folder structure:
- model_utils.py

This file contains all models I have tried
- data_utils.py

This contains preparing data from scratch (without using torchtext), including wrappers which generate data variable length data batches for training. 
- train_utils.py

This file contains the function used for training and calculating losses during training
- evaluation.py

This file contains calculation of f1 scores, downloaded from canvas. 
- submission.py

This file generate prediction.txt.
- preprocessing.ipynb

This is the notebook for preprocessing.

- run.ipynb
This should be the starting point. This is where I import all other functions from \*.py files and get results.



