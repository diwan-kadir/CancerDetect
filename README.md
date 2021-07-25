# CancerDetect

Team : Kadir Diwan | Rajeev Bandi | Arfah Upade 

Project : Detection of Cancer using Genome Profiles

Reference : git link of the Github repo we referred

# CancerDetect - Prediction & Detection of Cancer, based on gene profiles using Deep Learning

## I. INTRODUCTION
Cancer is the second leading cause of death globally with an estimated amount of 9.6 million deaths, or one in six deaths, estimated in 2018 by the World Health Organisation (WHO) [28]. A lot of the time, prevention of diseases like Cancer, Diabetes, and Alzheimers are treated late after their onset, as detection is done late.

>CancerDetect aims at giving medical professionals/researchers and patients/subjects a prediction model based on their current gene profiles that indicate the probability of the type of Cancer that they may be affected by. CancerDetect uses the patient’s gene profile to do the prediction.

## II. NEED FOR PROJECT
Diseases such as Cancer tend to be detected after their onset which leads to infeasible prevention and difficult or painful cures. If a prediction can be given to medical professionals and their patients about the probability of the patient having cancer, the medical professional may assist the patient in taking the right steps so as to detect and prevent the first and later stages of Cancer.

## III. METHODOLOGY

A. HOW DOES IT WORK?

CancerDetect provides a user-friendly interface to medical professionals, where they can upload the gene data of their patients after filling in the necessary details. A report is generated for the medical professional, in which the type of Cancer, the name of the Cancer, the probability percentage of the Cancer that could occur to the patient, and a remark such as “High” or “Low”, is given.

The report also presents a scatter plot for visualization purposes.
The report is printable and can be given to the
patient for further evaluation.
CancerDetect’s probability of the occurrence of
cancer within the human body works on a
prediction model based on concepts of
Bioinformatics, Deep Learning, and Gene
Profile Reading. This prediction model enables
CancerDetect to get the probability percentage
that is then given to the medical professional or
patient. [Refer to Section IV. PREDICTION
MODEL]
B. SOFTWARE USED
For building and training Deep learning models
various languages and their deep learning
specific libraries are available. Cancer detect
uses Python (version 3.8) and Google’s Keras
Library (version compatible with python 3.8)
along with other libraries such as Scikit learn,
Pickle, Pandas for data processing and other
miscellaneous jobs.
The frontend is hosted using Python’s Flask
framework which provides flexibility as well as
rapid development
Python 3.8, Keras, NumPy, Pandas, Pickle,
Scikit Learn, TensorFlow,
IDEs Used: Spyder, Jupyter Notebook
C. HARDWARE USED
The TCGA dataset consists of seven thousand
genes of eleven thousand individuals which in
total is roughly 600 MB and training the
convolutional model with 20 epochs would
definitely require external graphics. The model
was trained on a Graphics card having 4Gb
DDR5 memory and 7 Gbps memory speed
(Nvidia 1050 GPU). The prediction is not
GPU-dependent as compared to training.





[data1.csv](https://github.com/diwan-kadir/CancerDetect/files/6565741/data1.csv)

[data2.csv](https://github.com/diwan-kadir/CancerDetect/files/6565742/data2.csv)

[data3.csv](https://github.com/diwan-kadir/CancerDetect/files/6565743/data3.csv)

Accuracy Graph

![Acc](https://github.com/diwan-kadir/CancerDetect/blob/master/Accuracy.png)

Loss Graph

![Loss](https://github.com/diwan-kadir/CancerDetect/blob/master/Loss.png)
