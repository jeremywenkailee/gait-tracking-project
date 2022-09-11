# CMPT 353 Project Group JBJ
#### This Project has been transferred from the repository at SFU GitLab. This project was done in a collaborative effort - in the report PDF please note the contributions at the end of the document.

### Team Members
- Jeremy Lee, jwl45, 301308486
- Behrad Bakhshandeh, bbakhsha, 301367811
- Jiali Cai, jialic, 301372102

##### COMMAND TO RUN: python3 dataProcessing.py input predict output.csv > outputlog.txt

Data will run a comparison with scores against various models, choosing the model with the highest score for evaluation and fitting. Data will then run against test data to produce some test predictions. Test predictions will be compared to actual classification to produce an accuracy percentage.

**Please note that it does take the system roughly 30-45 seconds to run due to the nature of reading in the data.**

- input - folder to put the walking data. Data is already provided
- predict - folder to put the data for predicting. Data is already provided
- dataProcessing.py - program to read, process, and output results
- output.csv - output program for result comparison
- outputlog.txt - an outline of the processes in the program including Data reading/formating/processing, model testing, and accuracy evaluation

##### DEPENDANCIES AND LIBRARIES USED
- contextlib
- pandas.core.frame
- numpy
- pandas
- scipy
- sklearn.model_selection
- sklearn.naive_bayes
- sklearn.neighbors
- sklearn.ensemble
- sys
- glob
- os
- warnings
