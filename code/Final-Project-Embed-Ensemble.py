#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# In[1]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import random
df = pd.read_csv('DATA/matches2020.csv')
df


# In[2]:


df.dtypes
df = df.drop(columns=["Unnamed: 0","gameid","league","blueteam","redteam"])
df.count()


# In[3]:


df = df.dropna()
df


# ### Unique Blue Teams

# In[4]:



bluet = df[['bluetop','bluejungle','bluemid','blueadc','bluesupport']]
bluet

redt = df[['redtop','redjungle','redmid','redadc','redsupport']]
redt = redt.rename(columns={'redtop': 'bluetop', 'redjungle': 'bluejungle','redmid':'bluemid','redadc':'blueadc','redsupport':'bluesupport'})
allt = redt.append(bluet)
allt.groupby(['bluetop','bluejungle','bluemid','blueadc','bluesupport']).ngroups
allt


# ### Unique Red Teams

# In[5]:


df.groupby(['redtop','redjungle','redmid','redadc','redsupport']).ngroups


# In[6]:


df.groupby(['redtop','redjungle','redmid','redadc','redsupport']).ngroups


# ### Unique Champions

# In[7]:


topb = df['bluetop']
topr = df['redtop']
midb = df['bluemid']
midr = df['redmid']
jngb = df['bluejungle']
jngr = df['redjungle']
adcb = df['blueadc']
adcr = df['redadc']
supb = df['bluesupport']
supr = df['redsupport']
uni = df['redtop'].append(topb)
uni = uni.append(midb)
uni = uni.append(midr)
uni = uni.append(jngb)
uni = uni.append(jngr)
uni = uni.append(adcb)
uni = uni.append(adcr)
uni = uni.append(supb)
uni = uni.append(supr)
uni.nunique()
uni = uni.drop_duplicates()
uni


# ### Win Rate by Champion

# In[8]:


df


# In[9]:


bluewins = df.query('result == 1 & (bluetop == "Rumble" | bluejungle == "Rumble" | bluemid == "Rumble" | blueadc == "Rumble" | bluesupport == "Rumble")').count()


# In[10]:


redwins = df.query('result == 0 & (redtop == "Rumble" | redjungle == "Rumble" | redmid == "Rumble" | redadc == "Rumble" | redsupport == "Rumble")').count()


# In[11]:


bluelosses = df.query('result == 0 & (bluetop == "Rumble" | bluejungle == "Rumble" | bluemid == "Rumble" | blueadc == "Rumble" | bluesupport == "Rumble")').count()


# In[12]:


redlosses = df.query('result == 1 & (redtop == "Rumble" | redjungle == "Rumble" | redmid == "Rumble" | redadc == "Rumble" | redsupport == "Rumble")').count()


# In[13]:


winrate = (bluewins+redwins)/(bluelosses+redlosses)


# In[14]:


champs = []
winrates = []
totalplayed = []
for item in uni:
    champs.append(item)
    bluewins = df.query('result == 1 & (bluetop == "'+item+'" | bluejungle == "'+item+'" | bluemid == "'+item+'" | blueadc == "'+item+'" | bluesupport == "'+item+'")').count()
    redwins = df.query('result == 0 & (redtop == "'+item+'" | redjungle == "'+item+'" | redmid == "'+item+'" | redadc == "'+item+'" | redsupport == "'+item+'")').count()
    bluelosses = df.query('result == 0 & (bluetop == "'+item+'" | bluejungle == "'+item+'" | bluemid == "'+item+'" | blueadc == "'+item+'" | bluesupport == "'+item+'")').count()
    redlosses = df.query('result == 1 & (redtop == "'+item+'" | redjungle == "'+item+'" | redmid == "'+item+'" | redadc == "'+item+'" | redsupport == "'+item+'")').count()
    winrate = (bluewins+redwins)/(bluelosses+redlosses+bluewins+redwins)[0]
    winrates.append(winrate[0])
    totalplayed.append((bluelosses+redlosses+bluewins+redwins)[0])
    
print(winrates, totalplayed)


# In[15]:


ch_wr_gp = (champs, winrates, totalplayed)
d = {'champ': champs, 'winrate': winrates, 'totalplayed':totalplayed}
new = pd.DataFrame(data=d)
new = new.sort_values('totalplayed')


# In[16]:




ind = np.arange(len(new['champ']))  # the x locations for the groups
width = .4  # the width of the bars

plt.figure(figsize=(25,35))
plt.barh(ind, new['totalplayed'], width,
                color='violet', label='Games Played')

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Champion', size=20)
plt.xlabel('Games Played', size=20)
plt.title('Total Games Played By Champion & Winrate by Champion', size=20)
plt.yticks(ind, new['champ'])

for i, (p, pr) in enumerate(zip(new["winrate"], new["totalplayed"])):
    plt.text(s=('%.0f' % (p*100))+'%', x=(pr+20), y=i, color="black", verticalalignment="center", size=13)
    #plt.text(s=str(round(pr,0))+"%", x=pr-5, y=i, color="black",
             #verticalalignment="center", horizontalalignment="left", size=10)

plt.show()


# ### One Hot Encoding

# In[17]:


from sklearn.preprocessing import OneHotEncoder


# In[18]:


import tensorflow as tf
num_champions=150
champs_ = []
dic = {}
j=1
for col in df:
    if col != 'result':
        print(col)
        for i in range(len(df['bluetop'])):
            if(df[col][i] not in champs_):
                champs_.append(df[col][i])
                dic[df[col][i]] = j
                j = j + 1
                #print(df[col][i], j)
            

champs_.pop()
champs_.pop()
print(champs_)


# In[19]:


dic


# In[20]:



for col in df:
    if col != 'result':
        for i in range(len(df['bluetop'])):
            #print(df[col][i])
            df[col][i] = dic[df[col][i]]
            
df


# In[214]:


X = df


# ### Randomize SHuffle

# In[215]:


X=X.sample(frac=1)
X


# ### Sklearn

# In[216]:


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

results = X['result']
X = X.drop(axis=1,labels=['result'])


# In[217]:


X


# ### Tensorflow

# In[218]:


import tensorflow as tf
import sklearn


# In[219]:


r = random.randint(1, 100)

input_array = np.asarray(X.values).astype(np.float32)
target = np.asarray(results.values)

X_train, X_test, y_train, y_test = train_test_split(input_array, target, random_state=r)


# In[220]:


def fit_model():
    
    model = tf.keras.models.Sequential([tf.keras.layers.Embedding(148, 10, input_length=10),
                                   tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(300, activation='tanh', 
                                                         kernel_initializer='uniform', 
                                                         bias_initializer='zeros'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(300, activation='tanh', 
                                                         kernel_initializer='uniform', 
                                                         bias_initializer='zeros'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(300, activation='tanh', 
                                                         kernel_initializer='uniform', 
                                                         bias_initializer='zeros'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(300, activation='tanh', 
                                                         kernel_initializer='uniform', 
                                                         bias_initializer='zeros'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(1, activation='sigmoid')
                                   ])
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=7, validation_split=.05)
    return model


# In[221]:


# fit all models
n_members = 10
members = [fit_model() for _ in range(n_members)]


# In[222]:


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # argmax across classes
    result = np.round(summed/len(members))#np.argmax(summed, axis=1)
    #print(result)
    return result


# In[223]:


# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
    # select a subset of members
    subset = members[:n_members]
    # make prediction
    yhat = ensemble_predictions(subset, testX)
    # calculate accuracy
    return sklearn.metrics.accuracy_score(testy, yhat)


# In[224]:


# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores = list(), list()
for i in range(1, len(members)+1):
    # evaluate model with i members
    ensemble_score = evaluate_n_members(members, i, X_test, y_test)
    # evaluate the i'th model standalone
    #testy_enc = tf.keras.utils.to_categorical(y_test)
    _, single_score = members[i-1].evaluate(X_test, y_test, verbose=0)
    # summarize this step
    print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
    ensemble_scores.append(ensemble_score)
    single_scores.append(single_score)
# summarize average accuracy of a single final model
print('Accuracy %.3f (%.3f)' % (np.mean(single_scores), np.std(single_scores)))


# In[ ]:





# In[ ]:




