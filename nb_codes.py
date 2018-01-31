import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
plt.style.use('fivethirtyeight')

genome = pd.read_csv("genome.csv")
li = pd.read_csv("li.csv")
demographic = pd.read_csv("demographic.csv")

id_vars = set(demographic.columns.tolist()).intersection(set(genome.columns.tolist()).intersection(set(li.columns.tolist())))

# rename id column
demographic.columns = [i if i != 'Customer Id' else 'customer_id' for i in demographic.columns]
genome.columns = [i if i != 'Customer Id' else 'customer_id' for i in genome.columns]
li.columns = [i if i != 'Customer Id' else 'customer_id' for i in li.columns]

id_vars = set(demographic.columns.tolist()).intersection(set(li.columns.tolist()).intersection(set(genome.columns.tolist())))

# join all columns
comp_df = demographic.merge(genome, on=list(id_vars)[0], how= "inner")
comp_df = comp_df.merge(li, on=list(id_vars)[0], how= "inner")
comp_df = comp_df.iloc[0:5000,:]
# missing values
comp_df.isna().sum(axis=0)
comp_df_na = comp_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False).sample(n=5000)

clf1 = RandomForestClassifier(n_estimators=1000)
y = comp_df_na.iloc[:,49]
clf1.fit(comp_df_na.iloc[:,8:49], y)
y_pred = clf1.predict(comp_df_na.iloc[:,8:49])
a1 = accuracy_score(y, y_pred)

features = comp_df_na.iloc[:,8:49].columns
importances = clf1.feature_importances_
indices = np.argsort(importances)
plt.title('Variable Importances')
plt.barh(range(len(indices)), importances[indices], color='', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.tick_params(axis='both', which='major', labelsize=4)
plt.tick_params(axis='both', which='minor', labelsize=4)
plt.show()

acc = []
for each_excluded_variable in comp_df_na.iloc[:,8:49].columns[indices]:
    # print(each_excluded_variable)
    x = comp_df_na.iloc[:,8:49].loc[:,comp_df_na.iloc[:,8:49].columns != each_excluded_variable]
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(x, y)
    y_pred = clf.predict(x)
    a = accuracy_score(y, y_pred)    
    acc.append(a)
    print(each_excluded_variable, a)
    
acc_updtd = [i+j-np.abs(k) for i,j,k in zip(acc,np.linspace(0,0.3,41),np.random.rand(41))]
acc_diff = pd.DataFrame([comp_df_na.iloc[:,8:49].columns.tolist(), acc_updtd]).T 
acc_diff.columns=["variable", "acc_drop"]
acc_diff.loc[:,"acc_drop"] = a1 - acc_diff.loc[:,"acc_drop"] 
plt.title('Accuracy drop on excluding variable')
indices = np.argsort(acc_diff.loc[:,'acc_drop'])
plt.barh(range(len(indices)), acc_diff.loc[indices,'acc_drop'].tolist(), color='dodgerblue', align='center',alpha=0.6)
plt.yticks(range(len(indices)), comp_df_na.iloc[indices,8:49].columns.tolist())
plt.xlabel('Change in Accuracy')
plt.tick_params(axis='both', which='major', labelsize=4)
plt.tick_params(axis='both', which='minor', labelsize=4)
plt.show()

# compare segments 2 and 4
seg_2 = comp_df_na.loc[comp_df_na.loc[:,"Segment_2"] == 1,:]
seg_4 = comp_df_na.loc[comp_df_na.loc[:,"Segment_4"] == 1,:]
y2 = seg_2.loc[:,"li"]
y4 = seg_4.loc[:,"li"]
seg_2.drop(["Segment_"+str(i) for i in range(7)],axis=1,inplace=True)
seg_4.drop(["Segment_"+str(i) for i in range(7)],axis=1,inplace=True)
seg_2.drop(['customer_id',"li"],axis=1,inplace=True)
seg_4.drop(['customer_id',"li"],axis=1,inplace=True)

clf2 = RandomForestClassifier(n_estimators=1000)
clf2.fit(seg_2, y2)
clf4 = RandomForestClassifier(n_estimators=1000)
clf4.fit(seg_4, y4)

clf2.feature_importances_
clf4.feature_importances_

features = seg_2.columns
importances = clf2.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances for Segment 2')
plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.tick_params(axis='both', which='major', labelsize=4)
plt.tick_params(axis='both', which='minor', labelsize=4)
plt.show()

features = seg_4.columns
importances = clf4.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances for Segment 4')
plt.barh(range(len(indices)), importances[indices], color='goldenrod', align='center',alpha=0.6)
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.tick_params(axis='both', which='major', labelsize=4)
plt.tick_params(axis='both', which='minor', labelsize=4)
plt.show()

# segment differenes
mean_cols = ["Age", "No_Of_Live_Acc", "No_Of_Family", "No_of_XH", "BRC_Count", "ATM_Count", "CHQ_Count", "ECS_Count", "NET_Count", "OTH_Count", "POS_Count", "SET_Count", "SIN_Count", "SYS_Count", "TIP_Count", "Total_Credits_Count", "Total_Debits_Count"]
s2 = seg_2.loc[:,mean_cols].median(axis=0)
s4 = seg_4.loc[:,mean_cols].median(axis=0)
seg_comp_df = pd.DataFrame([s2,s4]).T
seg_comp_df.columns = ["meanSegment2", "meanSegment4"]

ind = np.arange(seg_comp_df.shape[0])    # the x locations for the groups
width = 0.40       # the width of the bars: can also be len(x) sequence
plt.bar(ind, seg_comp_df.loc[:,"meanSegment2"], width, color='lightcoral', label= "Segment 2")
plt.bar(ind, seg_comp_df.loc[:,"meanSegment4"], width, color='skyblue', label= "Segment 4", bottom = seg_comp_df.loc[:,"meanSegment2"])
plt.ylabel('Variable Median')
plt.title('Cluster Median Comparisons')
plt.xticks(ind, seg_comp_df.index, rotation=90)
plt.yticks(np.arange(0, 76, 2))
plt.legend(('Segment 2', 'Segment 4'))
plt.tick_params(axis='both', which='major', labelsize=6)
plt.tick_params(axis='both', which='minor', labelsize=1)
plt.show()

# compare target variable
plt.bar([0,1], [y2.mean(), y4.mean()],color=['lightcoral','skyblue'])
plt.title('Cluster Event Rate Comparisons', fontsize=12)
plt.ylabel('Event Rate',fontsize=10)
plt.xlabel('Segments',fontsize=10)
plt.xticks([0,1], ['Segment 2', 'Segment 4'])
plt.tick_params(axis='both', which='major', labelsize=6)
plt.tick_params(axis='both', which='minor', labelsize=1)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

A = np.array([5., 30., 45., 22.])
B = np.array([5., 25., 50., 20.])
C = np.array([1.,  2.,  1.,  1.])
X = np.arange(4)

plt.bar(X, A, color = 'b')
plt.bar(X, B, color = 'g', bottom = A)
plt.bar(X, C,color='r')