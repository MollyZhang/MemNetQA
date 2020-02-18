# CSE244_ML_for_NLP
Code for CSE244 winter 2020 homework 2: Slot Filling

### Staring script:
run.ipynb

### Report:
https://github.com/MollyZhang/CSE244_ML_for_NLP_HW2/blob/master/HW2%20report.pdf

### Folder structure:
- model_utils.py

This file contains all models I have tried
- data_utils.py

This contains preparing data with torchtext, for both ngram format and also word embedding format. This file also include wrappers which generate data batches for training. 
- train_utils.py

This file contains the function used for training and calculating losses during training
- evaluation.py

This file contains calculation of f1 scores
- submission.py

This file generate submission file for kaggle. It also includes an Ensemble class to combine results of multiple models into one final result. It also includes basic error anaylysis. 
- preprocessing.ipynb

This is the notebook for preprocessing:

- run.ipynb
This should be the starting point. This is where I import all other functions from \*.py files and get results.





# CSE244_ML_for_NLP_HW2
