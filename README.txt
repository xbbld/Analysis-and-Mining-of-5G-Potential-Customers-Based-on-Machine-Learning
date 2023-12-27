Analysis and Mining of 5G Potential Customers Based on Machine Learning
===========================================
Description
===========
With the continuous development and improvement of communication network engineering and new infrastructure technologies, China is gradually realizing the transition from a 4G society to a 5G society.So in order to improve the utilization rate of 5G users is required.So this item is goint to predict the dig out the users who may be the user of 5G. The process of building the prediction model mainly includes data pre-processing, feature engineering, training and evaluation of the model. Models were constructed based on the screened feature variables, including Random Forest model,and LightGBM model and the models are evaluated by AUC value index.Comparing two models we find out that LightGBM is more accurate and faster.
Example results
===========
	pricision	recall	f1-score	support
不是5G用户	0.88	0.84	0.86	2019
是5G用户	0.84	0.88	0.86	1982
accuracy			0.86	4001
macro avg	0.86	0.86	0.86	4001
weighted avg	0.86	0.86	0.86	4001
AUC指标为0.9227
	pricision	recall	f1-score	support
不是5G用户	0.86	0.81	0.83	2008
是5G用户	0.84	0.88	0.86	1993
accuracy			0.84	4001
macro avg	0.84	0.84	0.84	4001
weighted avg	0.84	0.84	0.84	4001
AUC指标为0.9120
Setup and configuration
=======================
All code for 5G_pred is written in Python and need a file named train.csv.
Firstly,download anaconda environments,
then download jupyter Notebook,
and download train.csv to 'C:\\Users\\asus\\Downloads\\train.csv' or you can update the code "train1 = pd.read_csv('C:\\Users\\asus\\Downloads\\train.csv')" to where you download the train.csv.
Finally run 5G_pred.py.
