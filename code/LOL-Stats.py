#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# In[182]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import random
df = pd.read_csv('DATA/matches2020.csv')
df


# In[183]:


df.dtypes
df = df.drop(columns=["Unnamed: 0","gameid","league","blueteam","redteam"])
df.count()


# In[184]:


df = df.dropna()
df


# ### Unique Blue Teams

# In[185]:



bluet = df[['bluetop','bluejungle','bluemid','blueadc','bluesupport']]
bluet

redt = df[['redtop','redjungle','redmid','redadc','redsupport']]
redt = redt.rename(columns={'redtop': 'bluetop', 'redjungle': 'bluejungle','redmid':'bluemid','redadc':'blueadc','redsupport':'bluesupport'})
allt = redt.append(bluet)
allt.groupby(['bluetop','bluejungle','bluemid','blueadc','bluesupport']).ngroups
allt


# ### Unique Red Teams

# In[186]:


df.groupby(['redtop','redjungle','redmid','redadc','redsupport']).ngroups


# In[187]:


df.groupby(['redtop','redjungle','redmid','redadc','redsupport']).ngroups


# ### Unique Champions

# In[188]:


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

# In[189]:


df


# In[190]:


bluewins = df.query('result == 1 & (bluetop == "Rumble" | bluejungle == "Rumble" | bluemid == "Rumble" | blueadc == "Rumble" | bluesupport == "Rumble")').count()


# In[191]:


redwins = df.query('result == 0 & (redtop == "Rumble" | redjungle == "Rumble" | redmid == "Rumble" | redadc == "Rumble" | redsupport == "Rumble")').count()


# In[192]:


bluelosses = df.query('result == 0 & (bluetop == "Rumble" | bluejungle == "Rumble" | bluemid == "Rumble" | blueadc == "Rumble" | bluesupport == "Rumble")').count()


# In[193]:


redlosses = df.query('result == 1 & (redtop == "Rumble" | redjungle == "Rumble" | redmid == "Rumble" | redadc == "Rumble" | redsupport == "Rumble")').count()


# In[194]:


winrate = (bluewins+redwins)/(bluelosses+redlosses)


# In[195]:


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


# In[196]:


ch_wr_gp = (champs, winrates, totalplayed)
d = {'champ': champs, 'winrate': winrates, 'totalplayed':totalplayed}
new = pd.DataFrame(data=d)
new = new.sort_values('totalplayed')


# In[197]:




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

# In[198]:


from sklearn.preprocessing import OneHotEncoder


# In[199]:


df


# Insert Stats

# In[200]:


df_stats = pd.read_csv('DATA/champion_stats.csv')
df_stats


# In[201]:


df_stats = df_stats.drop(axis=1,labels=['Unnamed: 0','key','tags','hp','hpperlevel','mp','mpperlevel','movespeed','armor','armorperlevel','spellblock','spellblockperlevel','attackrange','hpregen','hpregenperlevel','mpregen','mpregenperlevel','crit','critperlevel','attackdamage','attackdamageperlevel','attackspeed','attackspeedperlevel'])
df_stats


# create new columns for df

# In[202]:


df2 = df.copy()
df2.insert(0, 'bt_att', 'NA')
df2.insert(0, 'bt_def', 'NA')
df2.insert(0, 'bt_mag', 'NA')
df2.insert(0, 'bt_dif', 'NA')

df2.insert(0, 'rt_att', 'NA')
df2.insert(0, 'rt_def', 'NA')
df2.insert(0, 'rt_mag', 'NA')
df2.insert(0, 'rt_dif', 'NA')

df2.insert(0, 'bm_att', 'NA')
df2.insert(0, 'bm_def', 'NA')
df2.insert(0, 'bm_mag', 'NA')
df2.insert(0, 'bm_dif', 'NA')

df2.insert(0, 'rm_att', 'NA')
df2.insert(0, 'rm_def', 'NA')
df2.insert(0, 'rm_mag', 'NA')
df2.insert(0, 'rm_dif', 'NA')

df2.insert(0, 'bj_att', 'NA')
df2.insert(0, 'bj_def', 'NA')
df2.insert(0, 'bj_mag', 'NA')
df2.insert(0, 'bj_dif', 'NA')

df2.insert(0, 'rj_att', 'NA')
df2.insert(0, 'rj_def', 'NA')
df2.insert(0, 'rj_mag', 'NA')
df2.insert(0, 'rj_dif', 'NA')

df2.insert(0, 'bb_att', 'NA')
df2.insert(0, 'bb_def', 'NA')
df2.insert(0, 'bb_mag', 'NA')
df2.insert(0, 'bb_dif', 'NA')

df2.insert(0, 'rb_att', 'NA')
df2.insert(0, 'rb_def', 'NA')
df2.insert(0, 'rb_mag', 'NA')
df2.insert(0, 'rb_dif', 'NA')

df2.insert(0, 'bs_att', 'NA')
df2.insert(0, 'bs_def', 'NA')
df2.insert(0, 'bs_mag', 'NA')
df2.insert(0, 'bs_dif', 'NA')

df2.insert(0, 'rs_att', 'NA')
df2.insert(0, 'rs_def', 'NA')
df2.insert(0, 'rs_mag', 'NA')
df2.insert(0, 'rs_dif', 'NA')

df2


# In[203]:


df2


# In[204]:



import tensorflow as tf
for col in df:
    print(col)
    if col != 'result':
        for j in range(len(df_stats['id'])):
            for i in range(len(df['bluetop'])):
                if(df[col][i] == df_stats['id'][j]):
                    if(col == 'bluetop'):
                        df2.at[i, 'bt_att'] = df_stats['attack'][j]
                        df2.at[i, 'bt_def'] = df_stats['defense'][j]
                        df2.at[i, 'bt_mag'] = df_stats['magic'][j]
                        df2.at[i, 'bt_dif'] = df_stats['difficulty'][j]
                    elif(col == 'bluejungle'):
                        df2.at[i, 'bj_att'] = df_stats['attack'][j]
                        df2.at[i, 'bj_def'] = df_stats['defense'][j]
                        df2.at[i, 'bj_mag'] = df_stats['magic'][j]
                        df2.at[i, 'bj_dif'] = df_stats['difficulty'][j]
                    elif(col == 'bluemid'):
                        df2.at[i, 'bm_att'] = df_stats['attack'][j]
                        df2.at[i, 'bm_def'] = df_stats['defense'][j]
                        df2.at[i, 'bm_mag'] = df_stats['magic'][j]
                        df2.at[i, 'bm_dif'] = df_stats['difficulty'][j]
                    elif(col == 'blueadc'):
                        df2.at[i, 'bb_att'] = df_stats['attack'][j]
                        df2.at[i, 'bb_def'] = df_stats['defense'][j]
                        df2.at[i, 'bb_mag'] = df_stats['magic'][j]
                        df2.at[i, 'bb_dif'] = df_stats['difficulty'][j]
                    elif(col == 'bluesupport'):
                        df2.at[i, 'bs_att'] = df_stats['attack'][j]
                        df2.at[i, 'bs_def'] = df_stats['defense'][j]
                        df2.at[i, 'bs_mag'] = df_stats['magic'][j]
                        df2.at[i, 'bs_dif'] = df_stats['difficulty'][j]
                    
                    elif(col == 'redtop'):
                        df2.at[i, 'rt_att'] = df_stats['attack'][j]
                        df2.at[i, 'rt_def'] = df_stats['defense'][j]
                        df2.at[i, 'rt_mag'] = df_stats['magic'][j]
                        df2.at[i, 'rt_dif'] = df_stats['difficulty'][j]
                    elif(col == 'redjungle'):
                        df2.at[i, 'rj_att'] = df_stats['attack'][j]
                        df2.at[i, 'rj_def'] = df_stats['defense'][j]
                        df2.at[i, 'rj_mag'] = df_stats['magic'][j]
                        df2.at[i, 'rj_dif'] = df_stats['difficulty'][j]
                    elif(col == 'redmid'):
                        df2.at[i, 'rm_att'] = df_stats['attack'][j]
                        df2.at[i, 'rm_def'] = df_stats['defense'][j]
                        df2.at[i, 'rm_mag'] = df_stats['magic'][j]
                        df2.at[i, 'rm_dif'] = df_stats['difficulty'][j]
                    elif(col == 'redadc'):
                        df2.at[i, 'rb_att'] = df_stats['attack'][j]
                        df2.at[i, 'rb_def'] = df_stats['defense'][j]
                        df2.at[i, 'rb_mag'] = df_stats['magic'][j]
                        df2.at[i, 'rb_dif'] = df_stats['difficulty'][j]
                    elif(col == 'redsupport'):
                        df2.at[i, 'rs_att'] = df_stats['attack'][j]
                        df2.at[i, 'rs_def'] = df_stats['defense'][j]
                        df2.at[i, 'rs_mag'] = df_stats['magic'][j]
                        df2.at[i, 'rs_dif'] = df_stats['difficulty'][j]
            

df2


# In[205]:


df2


# In[206]:


df2 = df2.drop(axis=1,labels=['bluetop','redtop','bluejungle','redjungle','bluemid','redmid','blueadc','redadc','bluesupport','redsupport'])


# In[207]:


df2


# In[265]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[208]:


X = df2


# ### Randomize SHuffle

# In[209]:


X=X.sample(frac=1)
X


# In[210]:


len(X.columns)


# ### Sklearn

# In[211]:


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

results = X['result']
X = X.drop(axis=1,labels=['result'])


# In[216]:


X


# In[218]:


for col in X:
    for i in range(len(X['rs_dif'])):
        X.at[i, col] = X[col][i]/10


# In[219]:


X


# ### Tensorflow

# In[220]:


import tensorflow as tf


# In[258]:


#model_emb = tf.keras.models.Sequential([tf.keras.layers.Embedding(41, 3, input_length=41)])

model = tf.keras.models.Sequential([#tf.keras.layers.Embedding(41, 3, input_length=41),
                                   #tf.keras.layers.Flatten(),
                                    tf.keras.Input(shape=(40,)),
                                    tf.keras.layers.Dense(500, activation='relu', 
                                                         kernel_initializer='uniform'),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(500, activation='relu', 
                                                         kernel_initializer='uniform'),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(500, activation='relu', 
                                                         kernel_initializer='uniform'),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(1, activation='sigmoid')
                                   ])


# In[259]:


#model_emb.compile('rmsprop', 'mse')


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-1,
    decay_steps=10000,
    decay_rate=0.9)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[260]:


input_array = np.asarray(X.values).astype(np.float32)
target = np.asarray(results.values)
input_array


# In[261]:


#output_array = model_emb.predict(input_array)
#print(output_array.shape)
#output_array


# In[262]:


#X_train, X_test, y_train, y_test = train_test_split(output_array, results, random_state=19)
X_train, X_test, y_train, y_test = train_test_split(input_array, target, random_state=2)


# In[266]:


history = model.fit(X_train, y_train, epochs=20, validation_split=.2)


# In[264]:


model.evaluate(X_test, y_test)


# In[237]:


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




