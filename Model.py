import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
import matplotlib.pyplot as plt
from sklearn import preprocessing


def make(X_arr):
      X_arr=X_arr.to_numpy()
      X=np.zeros((X_arr.shape[0],3))
      for i in range(X_arr.shape[0]):     
#################pairs
            cnt=0
            idx=0
            arr = x1 = np.random.randint(10, size=5) 
            for j in range(1,10,2):
                  arr[idx]=X_arr[i][j]
                  idx+=1
                  for z in range(j+2,10,2):
                        if(X_arr[i][j]==X_arr[i][z]):
                              cnt+=1
            X[i][0]=cnt
################SEQUENCE
            flag=1
            arr.sort()
            for j in range(1,5):
                  if((arr[j]-arr[j-1])!=1):
                        flag=0
                        break
            if(flag):
                  X[i][1]=1
                  if(arr[0]==1 and arr[4]==13):
                        X[i][1]=2
################# 4 of a kind
            cnt=0
            for j in range(0,5):
                  if((arr[j]==arr[2])):
                        cnt+=1
            if(cnt==4):
                  X[i][0]=4
                  
######################FLUSH
            cnt=0
            for j in range(2,9,2):
                  if(X_arr[i][j]==X_arr[i][j-2]):
                        cnt+=1
            if(cnt==4):
                  X[i][2]=1 
      print("DONE") 
      X= preprocessing.scale(X) 
      return X

train_data_path = "train.csv"
train_data=pd.read_csv(train_data_path)

X_train, X_val = train_test_split(train_data, test_size=0.2,random_state=42)
X_train,y_train = X_train.iloc[:,:-1],X_train.iloc[:,-1]
X_val,y_val = X_val.iloc[:,:-1],X_val.iloc[:,-1]


clf = LogisticRegression(random_state=0,max_iter=150)
clf.fit(make(X_train),y_train)
y_pred = clf.predict(make(X_val))
precision = precision_score(y_val,y_pred,average='micro')
recall = recall_score(y_val,y_pred,average='micro')
accuracy = accuracy_score(y_val,y_pred)
f1 = f1_score(y_val,y_pred,average='macro')
print("Accuracy of the model is :" ,accuracy)
print("Recall of the model is :" ,recall)
print("Precision of the model is :" ,precision)
print("F1 score of the model is :" ,f1)

final_test_path = "test.csv"
final_test = pd.read_csv(final_test_path)
submission = clf.predict(make(final_test))
submission = pd.DataFrame(submission)
submission.to_csv('/tmp/submission.csv',header=['label'],index=False)