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



bluet = df[['bluetop','bluejungle','bluemid','blueadc','bluesupport','result']]
bluet

redt = df[['redtop','redjungle','redmid','redadc','redsupport','result']]
redt = redt.rename(columns={'redtop': 'bluetop', 'redjungle': 'bluejungle','redmid':'bluemid','redadc':'blueadc','redsupport':'bluesupport'})
allt = redt.append(bluet)
allt.groupby(['bluetop','bluejungle','bluemid','blueadc','bluesupport']).ngroups
allt


# In[5]:


print(len(allt))


# ### Unique Red Teams

# In[6]:


allt.groupby(['redtop','redjungle','redmid','redadc','redsupport']).ngroups


# In[6]:


allt.groupby(['redtop','redjungle','redmid','redadc','redsupport']).ngroups


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


# In[243]:




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

# In[16]:


allt = allt.reset_index()
allt = allt.drop(columns=['index'])
allt


# In[244]:


from sklearn.preprocessing import OneHotEncoder


# In[17]:


import tensorflow as tf
num_champions=150
champs_ = []
dic = {}
j=1
for col in allt:
    if col != 'result':
        print(col)
        for i in range(len(allt['bluetop'])):
            if(allt[col][i] not in champs_):
                champs_.append(allt[col][i])
                dic[allt[col][i]] = j
                j = j + 1
                #print(df[col][i], j)
            

champs_.pop()
champs_.pop()
print(champs_)


# In[18]:


dic


# In[19]:



for col in allt:
    if col != 'result':
        for i in range(len(allt['bluetop'])):
            #print(df[col][i])
            allt.at[i, col] = dic[allt[col][i]]
            
allt


# In[20]:


X = allt


# ### Randomize SHuffle

# In[21]:


X=X.sample(frac=1)
X


# ### Sklearn

# In[22]:


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

results = X['result']
X = X.drop(axis=1,labels=['result'])


# In[23]:


X


# ### Tensorflow

# In[24]:


import tensorflow as tf


# In[46]:


model_emb = tf.keras.models.Sequential([tf.keras.layers.Embedding(148, 10, input_length=5)])

model = tf.keras.models.Sequential([tf.keras.layers.Embedding(148, 10, input_length=5),
                                   tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(50, activation='relu'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(50, activation='relu'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(50, activation='relu'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(50, activation='relu'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(1, activation='sigmoid')
                                   ])


# In[47]:


model_emb.compile('rmsprop', 'mse')


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-1,
    decay_steps=10000,
    decay_rate=0.9)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[48]:


input_array = np.asarray(X.values).astype(np.float32)
target = np.asarray(results.values)
target


# In[49]:


output_array = model_emb.predict(input_array)
print(output_array.shape)
output_array


# In[51]:


#X_train, X_test, y_train, y_test = train_test_split(output_array, results, random_state=19)
X_train, X_test, y_train, y_test = train_test_split(input_array, target, random_state=2)


# In[52]:


history = model.fit(X_train, y_train, epochs=10, validation_split=.2)


# In[53]:


model.evaluate(X_test, y_test)


# In[32]:


print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[358]:


predictions = model.predict(X_test)
predictions


# In[197]:


y_test


# In[ ]:




