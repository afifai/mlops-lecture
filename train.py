import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# DATA PREPRARATION
df = pd.read_csv('data/bank-full.csv', delimiter=';')

# Drop Columns
df = df.drop(columns = ['job', 'marital', 'education','contact', 'month', 'poutcome'])

# mapping features
df['default'] = df['default'].map( 
                   {'yes':1 ,'no':0}) 
df['housing'] = df['housing'].map( 
                   {'yes':1 ,'no':0}) 
df['loan'] = df['loan'].map( 
                   {'yes':1 ,'no':0}) 

# split dataset
X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

# MODELLING
model = RandomForestClassifier(n_estimators=100, random_state=46)
model.fit(X_train, y_train)

# Model Report
train_score = model.score(X_train, y_train) * 100
test_score = model.score(X_test, y_test) * 100

# Create metrics score file as txt
with open('metrics.txt', 'w') as f:
    f.write("Training Accuracy: {}\n".format(train_score))
    f.write("Testing Accuracy: {}\n".format(test_score))

# Feature Importance Plot
importances = model.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"])
feature_df = feature_df.sort_values('importance', ascending=False)

axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Importance',fontsize = axis_fs) 
ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png",dpi=120) 
plt.close()

# Confusion Matrix Plot
axis_fs = 18 #fontsize
title_fs = 22 #fontsize

cm = confusion_matrix(y_test, model.predict(X_test))
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
ax.set_xlabel('Predicted',fontsize = axis_fs)
ax.set_ylabel('True', fontsize = axis_fs)
ax.set_title('Confusion Matrix', fontsize = title_fs)

plt.tight_layout()
plt.savefig("confusion_matrix.png",dpi=120) 
plt.close()