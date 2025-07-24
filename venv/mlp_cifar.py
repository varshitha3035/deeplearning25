import libraries
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#import data
(X_train,y_train),(X_test,y_test)=cifar10.load_data()

#categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
 
#buid the architecture
model=Sequential()
model.add(Flatten(input_shape=(32,32,3)))
