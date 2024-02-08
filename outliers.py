


#Q1. The nature of the dataset is really important here. An outliner could be a sensor's failure as well as
#    a particular value that one want to observe, like a disturbance in the ground anouncing an earthquake.
#    But it must be aprehended.

#    In the first case : we don't want a AI using supervised learning to learn on false values, aswell as we dont want
#    our unsupervised AI interpretting in its own sense a false value and making something out of it. But,
#    if the unsupervised one automatically detects failures, and knows those are failures, it could be great (sounds hard though).
#    In the second case : we always want to get those values, and always try to make something out of it, for both
#    supervised and unsupervised learning.

#Q2. a) I know absolutely nothing about the proportion, and that's the difficulty. To remediate, I can do reasearches about
#    the data I'm manipulating. But generally, by definition, outliners shouldn't exceed 1 to 5% in my opinion.

#    b) By assuming their proportion is low enough, when can look at the mean and standard deviation of our data set, and
#    pop out any value greater than maybe 3 or 5 times the standard deviation, relatively to the mean. It's basically a filter.

#    c) Statistical data, for example the law that the data is following, its usual parameters, and the method used to acquire them,
#    with the related errors of measure. I also really would like to have a graph of those data.

#Q3

import matplotlib.pyplot as plt
import numpy as np

import scipy as sp
mat_data = sp.io.loadmat('outlier_data.mat')
X = mat_data["X"]

# plt.scatter(X[:, 0], X[:, 1], color='b')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.title('oultlier data')
# plt.grid(True)
# plt.show()


#Q4 : They are the points outside the group in the middle, they are 6 of them.

#print(6/len(X))

# Proportion : ~2% which is okay

#Q5 : just cut out the one too far from the average using mean and std on both features

def outliner_cutter(data_to_clean):
    outliner_list = []

    X1_avg = 0
    X2_avg = 0
    n = len(X)

    #average x and average y
    for e in data_to_clean:
        X1_avg += e[0]/n
        X2_avg += e[1]/n

    #std in x and std in y

    sigX1 = 0
    sigX2 = 0

    for e in data_to_clean:
        sigX1 += (e[0]-X1_avg)**2
        sigX2 += (e[1] - X2_avg)**2

    sigX1 = (sigX1/n)**(0.5)
    sigX2 = (sigX2/n)**(0.5)

    #detection

    for e in data_to_clean:
        checkX1 = abs(X1_avg-e[0])
        checkX2 = abs(X2_avg-e[1])

        if (checkX1 > 3*sigX1 or checkX2 > 3*sigX2):
            outliner_list.append(e)

    return outliner_list



#print(outliner_list) : we got 6 values as expected !

#This is a very simple-to-implement quick method, pretty strong for large batches of data beacuse of the Central-Limit Theorem,
#But on smaller or weirder datasets it becomes completely irrelevent (small batches, data following a particular law, ...)
#A better way to do that is to detect first the law from X's features, then see which components arn't following it
#I also really like method using the "derivating", detecting outliners by a sudden derivative dirac when looking at one
#feature at a time like it was a function. There are fast and robst algorithms that detects such peaks on a given function,
#don't remember the name but a friend worked on it and it was pretty impressive.

#Q6.

outliners = np.array(outliner_cutter(X))

# plt.scatter(X[:, 0], X[:, 1], color='b', label='data')
# plt.scatter(outliners[:, 0], outliners[:, 1], color='r', label='outliners')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.title('oultlier data')
# plt.grid(True)
# plt.legend()
# plt.show()

import pandas as pd
import pandas_datareader.data as web
import datetime
start = datetime.datetime(2021, 6, 1)
end = datetime.datetime.today()
sign = "NVDA"
string = "stooq"
df = web.DataReader(sign, string, start=start, end=end)

#Q7.

# plt.scatter(df['Close'], df['Volume'], color='b')
# plt.title('Volume and Close')
# plt.xlabel('Close')
# plt.ylabel('Volume')
# plt.legend()
# plt.grid(True)
# plt.show()

#Q8. The dates that you asked to retrives data of aren't valid, so i changed the start above Q7
#to 2021 in order to get them all

start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2022-12-31')
subset_df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
data_set1 = list(zip(subset_df['Volume'], subset_df['Close']))
out_list1 = np.array(outliner_cutter(data_set1))
data_set1 = np.array(data_set1)

start_date = pd.to_datetime('2023-01-01')
end_date = pd.to_datetime('2023-06-30')
subset_df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
data_set2 = list(zip(subset_df['Volume'], subset_df['Close']))
out_list2 = np.array(outliner_cutter(data_set2))
data_set2 = np.array(data_set2)

start_date = pd.to_datetime('2021-01-01')
end_date = pd.to_datetime('2022-12-31')
subset_df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
data_set3 = list(zip(subset_df['Volume'], subset_df['Close']))
out_list3 = np.array(outliner_cutter(data_set3))
data_set3 = np.array(data_set3)

#here you just have to change the number 1 to 2 or 3 if you want to check the corresponding sets
def plot(i):
  if i==3:
    plt.scatter(data_set3[:, 0], data_set3[:, 1], color='r', label='normal data')
    plt.scatter(out_list3[:,0], out_list3[:,1], color='b', label='outliners')
  if i == 2:
    plt.scatter(data_set2[:, 0], data_set2[:, 1], color='r', label='normal data')
    plt.scatter(out_list2[:,0], out_list2[:,1], color='b', label='outliners')
  if i==1:
    plt.scatter(data_set1[:, 0], data_set1[:, 1], color='r', label='normal data')
    plt.scatter(out_list1[:,0], out_list1[:,1], color='b', label='outliners')
    plt.title('Volume and Close')
  plt.xlabel('Close')
  plt.ylabel('Volume')
  plt.legend()
  plt.grid(True)
  plt.show()


#plot(1)
  


#The result is not bad, but 1: i don't manage when the outliner's list is empty (not hard though),
#and 2: clearly some points should be in the list of outliner's and they aren't,
#especially those with too much volumes

#Q9. to be cleaner i could put all of this in a single method with given parameters : number of plot
#desired, dimension, srtating date and end date, and the rest is automatic.
#I could also apply several different filters, and check if they find the same outlining values.

#Q10. Couldn't test if it works, not enough time + i don't have colab pro
#but hope it does !
