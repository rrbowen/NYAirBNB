import pandas
import matplotlib
import numpy as numpy
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics
from matplotlib import pyplot
import math
import random

print("hey now coders")
airbnb = pandas.read_csv("AB_NYC_2019_reduced.csv")
for col in airbnb.columns: 
    print(col)
airbnb.info()
#reshuffle the dataset, so kfold wont just draw in order
random.seed(214)
airbnb = airbnb.sample(frac=1).reset_index(drop=True)


borough_dummies = pandas.get_dummies(airbnb['borough'])
print(borough_dummies)
room_dummies = pandas.get_dummies(airbnb['room_type'])
print(room_dummies)
#add experience squared
#exp2 = dataset['exp'].astype(int)**2
data = pandas.concat([
	airbnb[['min_nights','num_reviews','host_count']],
	borough_dummies,
	room_dummies,
	], axis=1).values
shaped = data.shape
print(shaped)


target = airbnb.iloc[:,0].values
shapet = target.shape
print(shapet)



kfold_object = KFold(n_splits = 4)
kfold_object.get_n_splits(airbnb)
#print(kfold_object)


#### k fold training of the linear model
i = 0
for training_index, test_index in kfold_object.split(airbnb):
	print(i)
	i =i+1
	#test_case = test_case+1
	numpy.set_printoptions(suppress=True)
	print("test: ",test_index)
	data_test = data[test_index]
	data_training = data[training_index]
	target_training = target[training_index]
	target_test = target[test_index]
	machine = linear_model.LinearRegression()
	machine.fit(data_training,target_training)
	new_target = machine.predict(data_test)
	numpy.set_printoptions(precision=3)
	print("R2 score: ",metrics.r2_score(target_test,new_target))
	print(machine.coef_)

#### k fold training of the lasso model


i = 0
for training_index, test_index in kfold_object.split(airbnb):
	print(i)
	i =i+1
	#test_case = test_case+1
	numpy.set_printoptions(suppress=True)
	print("test: ",test_index)
	data_test = data[test_index]
	data_training = data[training_index]
	target_training = target[training_index]
	target_test = target[test_index]
	machine = linear_model.Lasso(alpha=.01, normalize = True)
	machine.fit(data_training,target_training)
	new_target = machine.predict(data_test)
	numpy.set_printoptions(precision=3)
	print("R2 score: ",metrics.r2_score(target_test,new_target))
	print(machine.coef_)



