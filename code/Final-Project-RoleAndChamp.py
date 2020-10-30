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


# ### Load Hashmap

# In[4]:


df_c = pd.read_csv('DATA/champion_stats.csv')

champkeys = {}

for i in range(len(df_c["key"])):
    champkeys[df_c["key"][i]] = df_c["id"][i]

champkeys


# ### Ranked Data

# In[5]:


md = pd.read_csv('DATA/match_data_version1.csv')
md


# In[6]:


df_r = df.copy()
for i in range(len(md["gameMode"])):
    if(i%1000 == 0):
        print(i)
        
    if(md["gameMode"][i] == "CLASSIC"):
        
        # get the champions
        next_index = len(df['bluetop'])
        #print(next_index)
        string = md["participants"][i]
        loc = 0
        loc_p = 0
        loc_l = 0
        loc_w = 0
        
        bt_f = False
        bm_f = False
        bs_f = False
        bb_f = False
        bj_f = False
        
        rt_f = False
        rm_f = False
        rs_f = False
        rb_f = False
        rj_f = False
        
        for p in range(10):
            
            #get win
            if(p == 0):
                loc_w = string.find("win", (loc_w+1), len(string)-1)
                won = string[loc_w+6:loc_w+10]
                bwin = 0
                if(won.lower() == "true"):
                    bwin = 1
            
            #get champ id
            loc = string.find("championId", (loc+1), len(string)-1)
            space = string.find(",", (loc+13), len(string)-1)
            try:
                champid = int(string[loc+13:space])
            except ValueError as err:
                print("err")
            
            #get position
            loc_p = string.find("role", (loc_p+1), len(string)-1)
            support = string[loc_p+12:loc_p+19]
            loc_l = string.find("lane", (loc_l+1), len(string)-1)
            lane = string[loc_l+8]
            #print(lane)
            #champid = string[loc+13:space]
            
            if(p < 5 and support == "SUPPORT"):
                df.at[next_index, "bluesupport"] = champkeys[champid]
                #print("bS: ", champkeys[champid])
                
            elif(p < 5 and support != "SUPPORT"):
                if(p < 5 and lane == "B"):
                    df.at[next_index, "blueadc"] = champkeys[champid]
                    #print("bB: ", champkeys[champid])
                elif(p < 5 and lane == "J"):
                    df.at[next_index, "bluejungle"] = champkeys[champid]
                    #print("bJ: ", champkeys[champid])
                elif(p < 5 and lane == "M"):
                    df.at[next_index, "bluemid"] = champkeys[champid]
                    #print("bM: ", champkeys[champid])
                elif(p < 5 and lane == "T"):
                    df.at[next_index, "bluetop"] = champkeys[champid]
                    #print("bT: ", champkeys[champid])
                
            if(p >= 5 and support == "SUPPORT"):
                df.at[next_index, "redsupport"] = champkeys[champid]
                #print("rS: ", champkeys[champid])
                
            elif(p >= 5 and support != "SUPPORT"):
                if(p >= 5 and lane == "B"):
                    df.at[next_index, "redadc"] = champkeys[champid]
                    #print("rB: ", champkeys[champid])
                elif(p >= 5 and lane == "J"):
                    df.at[next_index, "redjungle"] = champkeys[champid]
                    #print("rJ: ", champkeys[champid])
                elif(p >= 5 and lane == "M"):
                    df.at[next_index, "redmid"] = champkeys[champid]
                    #print("rM: ", champkeys[champid])
                elif(p >= 5 and lane == "T"):
                    df.at[next_index, "redtop"] = champkeys[champid]
                    #print("rT: ", champkeys[champid])
                
            #print(next_index)
                
            if(p == 0):
                df.at[next_index, "result"] = bwin
                #print("result: ", bwin)
            
        
        #break
            
            
    
#print(md["participants"][0])


# In[7]:


print(len(df["bluetop"]))
df = df.dropna()
print(len(df["bluetop"]))
df


# ### Unique Blue Teams

# In[8]:



bluet = df[['bluetop','bluejungle','bluemid','blueadc','bluesupport']]
bluet

redt = df[['redtop','redjungle','redmid','redadc','redsupport']]
redt = redt.rename(columns={'redtop': 'bluetop', 'redjungle': 'bluejungle','redmid':'bluemid','redadc':'blueadc','redsupport':'bluesupport'})
allt = redt.append(bluet)
allt.groupby(['bluetop','bluejungle','bluemid','blueadc','bluesupport']).ngroups
allt


# ### Unique Red Teams

# In[9]:


df.groupby(['redtop','redjungle','redmid','redadc','redsupport']).ngroups


# In[10]:


df.groupby(['redtop','redjungle','redmid','redadc','redsupport']).ngroups


# ### Unique Champions

# In[11]:


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

# In[12]:


df


# In[13]:


bluewins = df.query('result == 1 & (bluetop == "Rumble" | bluejungle == "Rumble" | bluemid == "Rumble" | blueadc == "Rumble" | bluesupport == "Rumble")').count()


# In[14]:


redwins = df.query('result == 0 & (redtop == "Rumble" | redjungle == "Rumble" | redmid == "Rumble" | redadc == "Rumble" | redsupport == "Rumble")').count()


# In[15]:


bluelosses = df.query('result == 0 & (bluetop == "Rumble" | bluejungle == "Rumble" | bluemid == "Rumble" | blueadc == "Rumble" | bluesupport == "Rumble")').count()


# In[16]:


redlosses = df.query('result == 1 & (redtop == "Rumble" | redjungle == "Rumble" | redmid == "Rumble" | redadc == "Rumble" | redsupport == "Rumble")').count()


# In[17]:


winrate = (bluewins+redwins)/(bluelosses+redlosses)


# In[18]:


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


# In[19]:


ch_wr_gp = (champs, winrates, totalplayed)
d = {'champ': champs, 'winrate': winrates, 'totalplayed':totalplayed}
new = pd.DataFrame(data=d)
new = new.sort_values('totalplayed')


# In[20]:




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


# In[23]:


df = df.reset_index()
df = df.drop(axis=1, labels=['index'])
df


# In[ ]:


df_copy = df.copy()


# ### One Hot Encoding

# In[21]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


df.insert(0, 'bt_class', 'NA')
df.insert(0, 'bt_subclass', 'NA')
df.insert(0, 'rt_class', 'NA')
df.insert(0, 'rt_subclass', 'NA')
df.insert(0, 'bm_class', 'NA')
df.insert(0, 'bm_subclass', 'NA')
df.insert(0, 'rm_class', 'NA')
df.insert(0, 'rm_subclass', 'NA')
df.insert(0, 'bj_class', 'NA')
df.insert(0, 'bj_subclass', 'NA')
df.insert(0, 'rj_class', 'NA')
df.insert(0, 'rj_subclass', 'NA')
df.insert(0, 'bb_class', 'NA')
df.insert(0, 'bb_subclass', 'NA')
df.insert(0, 'rb_class', 'NA')
df.insert(0, 'rb_subclass', 'NA')
df.insert(0, 'bs_class', 'NA')
df.insert(0, 'bs_subclass', 'NA')
df.insert(0, 'rs_class', 'NA')
df.insert(0, 'rs_subclass', 'NA')


# In[24]:



for col in df:
    if (col != 'result' and col != 'bt_class' and
       col != 'bt_subclass' and col != 'rt_class' and 
        col != 'rt_subclass' and col != 'result' and 
        col != 'bm_subclass' and col != 'bm_class' and 
        col != 'rm_subclass' and col != 'rm_class' and 
        col != 'bj_subclass' and col != 'bj_class' and 
        col != 'rj_subclass' and col != 'rj_class' and 
        col != 'bb_subclass' and col != 'bb_class' and 
        col != 'rb_subclass' and col != 'rb_class' and 
        col != 'bs_subclass' and col != 'bs_class' and 
        col != 'rs_subclass' and col != 'rs_class'):
            
        for i in range(len(df['bluetop'])): 
            
            if(i%1000==0):
                print(i)
            
            if(df[col][i] == 'Bard' or df[col][i] == "Morgana" or
              df[col][i] == "Blitzcrank" or df[col][i] == "Neeko" or
              df[col][i] == "Ivern" or df[col][i] == 'Rakan' or
              df[col][i] == "Jhin" or df[col][i] == "Thresh" or
              df[col][i] == "Lux" or df[col][i] == "Zyra"):
                if col == 'bluetop':
                    df.at[i, "bt_class"] = "Controller"
                    df.at[i, "bt_subclass"] = "Catcher"
                elif col == 'bluemid':
                    df.at[i, "bm_class"] = "Controller"
                    df.at[i, "bm_subclass"] = "Catcher"
                elif col == 'bluejungle':
                    df.at[i, "bj_class"] = "Controller"
                    df.at[i, "bj_subclass"] = "Catcher"
                elif col == 'blueadc':
                    df.at[i, "bb_class"] = "Controller"
                    df.at[i, "bb_subclass"] = "Catcher"
                elif col == 'bluesupport':
                    df.at[i, "bs_class"] = "Controller"
                    df.at[i, "bs_subclass"] = "Catcher"
                elif col == 'redtop':
                    df.at[i, "rt_class"] = "Controller"
                    df.at[i, "rt_subclass"]= "Catcher"
                elif col == 'redmid':
                    df.at[i, "rm_class"] = "Controller"
                    df.at[i, "rm_subclass"] = "Catcher"
                elif col == 'redjungle':
                    df.at[i, "rj_class"] = "Controller"
                    df.at[i, "rj_subclass"] = "Catcher"
                elif col == 'redadc':
                    df.at[i, "rb_class"] = "Controller"
                    df.at[i, "rb_subclass"] = "Catcher"
                elif col == 'redsupport':
                    df.at[i, "rs_class"] = "Controller"
                    df.at[i, "rs_subclass"] = "Catcher"
            
            
                
            elif(df[col][i] == "Janna" or df[col][i] == "Seraphine" or
              df[col][i] == "Karma" or df[col][i] == "Sona" or
              df[col][i] == "Lulu" or df[col][i] == "Soraka" or
              df[col][i] == "Nami" or df[col][i] == "Taric" or
              df[col][i] == "Senna" or df[col][i] == "Yuumi"):
                if col == 'bluetop':
                    df.at[i, "bt_class"]= "Controller"
                    df.at[i, "bt_subclass"] = "Enchanter"
                elif col == 'bluemid':
                    df.at[i, "bm_class"] = "Controller"
                    df.at[i, "bm_subclass"] = "Enchanter"
                elif col == 'bluejungle':
                    df.at[i, "bj_class"] = "Controller"
                    df.at[i, "bj_subclass"] = "Enchanter"
                elif col == 'blueadc':
                    df.at[i, "bb_class"] = "Controller"
                    df.at[i, "bb_subclass"] = "Enchanter"
                elif col == 'bluesupport':
                    df.at[i, "bs_class"] = "Controller"
                    df.at[i, "bs_subclass"] = "Enchanter"
                elif col == 'redtop':
                    df.at[i, "rt_class"] = "Controller"
                    df.at[i, "rt_subclass"] = "Enchanter"
                elif col == 'redmid':
                    df.at[i, "rm_class"] = "Controller"
                    df.at[i, "rm_subclass"]= "Enchanter"
                elif col == 'redjungle':
                    df.at[i, "rj_class"] = "Controller"
                    df.at[i, "rj_subclass"] = "Enchanter"
                elif col == 'redadc':
                    df.at[i, "rb_class"] = "Controller"
                    df.at[i, "rb_subclass"] = "Enchanter"
                elif col == 'redsupport':
                    df.at[i, "rs_class"] = "Controller"
                    df.at[i, "rs_subclass"] = "Enchanter"
                
            elif(df[col][i] == "Camille" or df[col][i] == "Pantheon" or
              df[col][i] == "Diana" or df[col][i] == "RekSai" or
              df[col][i] == "Elise" or df[col][i] == "Renekton" or
              df[col][i] == "Hecarim" or df[col][i] == "Rengar" or
              df[col][i] == "Irelia" or df[col][i] == "Skarner" or
              df[col][i] == "JarvanIV" or df[col][i] == "Vi" or
              df[col][i] == "Keyn" or df[col][i] == "Warwick" or
              df[col][i] == "Kled" or df[col][i] == "MonkeyKing" or
              df[col][i] == "LeeSin" or df[col][i] == "XinZhao" or
              df[col][i] == "Olaf"):
                if col == 'bluetop':
                    df.at[i, "bt_class"] = "Fighter"
                    df.at[i, "bt_subclass"] = "Diver"
                elif col == 'bluemid':
                    df.at[i, "bm_class"] = "Fighter"
                    df.at[i, "bm_subclass"] = "Diver"
                elif col == 'bluejungle':
                    df.at[i, "bj_class"] = "Fighter"
                    df.at[i, "bj_subclass"] = "Diver"
                elif col == 'blueadc':
                    df.at[i, "bb_class"] = "Fighter"
                    df.at[i, "bb_subclass"] = "Diver"
                elif col == 'bluesupport':
                    df.at[i, "bs_class"] = "Fighter"
                    df.at[i, "bs_subclass"] = "Diver"
                elif col == 'redtop':
                    df.at[i, "rt_class"] = "Fighter"
                    df.at[i, "rt_subclass"] = "Diver"
                elif col == 'redmid':
                    df.at[i, "rm_class"] = "Fighter"
                    df.at[i, "rm_subclass"] = "Diver"
                elif col == 'redjungle':
                    df.at[i, "rj_class"] = "Fighter"
                    df.at[i, "rj_subclass"] = "Diver"
                elif col == 'redadc':
                    df.at[i, "rb_class"] = "Fighter"
                    df.at[i, "rb_subclass"] = "Diver"
                elif col == 'redsupport':
                    df.at[i, "rs_class"] = "Fighter"
                    df.at[i, "rs_subclass"]= "Diver"
            
            elif(df[col][i] == "Aatrox" or df[col][i] == "Sett" or
              df[col][i] == "Darius" or df[col][i] == "Shyvana" or
              df[col][i] == "DrMundo" or df[col][i] == "Trundle" or
              df[col][i] == "Garen" or df[col][i] == "Udyr" or
              df[col][i] == "Illaoi" or df[col][i] == "Urgot" or
              df[col][i] == "Mordekaiser" or df[col][i] == "Volibear" or
              df[col][i] == "Nasus" or df[col][i] == "Yorick"):
                if col == 'bluetop':
                    df.at[i, "bt_class"] = "Fighter"
                    df.at[i, "bt_subclass"] = "Juggernaut"
                elif col == 'bluemid':
                    df.at[i, "bm_class"] = "Fighter"
                    df.at[i, "bm_subclass"] = "Juggernaut"
                elif col == 'bluejungle':
                    df.at[i, "bj_class"] = "Fighter"
                    df.at[i, "bj_subclass"] = "Juggernaut"
                elif col == 'blueadc':
                    df.at[i, "bb_class"] = "Fighter"
                    df.at[i, "bb_subclass"] = "Juggernaut"
                elif col == 'bluesupport':
                    df.at[i, "bs_class"] = "Fighter"
                    df.at[i, "bs_subclass"] = "Juggernaut"
                elif col == 'redtop':
                    df.at[i, "rt_class"] = "Fighter"
                    df.at[i, "rt_subclass"] = "Juggernaut"
                elif col == 'redmid':
                    df.at[i, "rm_class"] = "Fighter"
                    df.at[i, "rm_subclass"] = "Juggernaut"
                elif col == 'redjungle':
                    df.at[i, "rj_class"] = "Fighter"
                    df.at[i, "rj_subclass"] = "Juggernaut"
                elif col == 'redadc':
                    df.at[i, "rb_class"] = "Fighter"
                    df.at[i, "rb_subclass"] = "Juggernaut"
                elif col == 'redsupport':
                    df.at[i, "rs_class"] = "Fighter"
                    df.at[i, "rs_subclass"] = "Juggernaut"
            
            elif(df[col][i] == "Jayce" or df[col][i] == "Xerath" or
              df[col][i] == "Lux" or df[col][i] == "Ziggs" or
              df[col][i] == "Varus" or df[col][i] == "Zoe" or
              df[col][i] == "Velkoz"):
                if col == 'bluetop':
                    df.at[i, "bt_class"] = "Mage"
                    df.at[i, "bt_subclass"] = "Artillery"
                elif col == 'bluemid':
                    df.at[i, "bm_class"] = "Mage"
                    df.at[i, "bm_subclass"] = "Artillery"
                elif col == 'bluejungle':
                    df.at[i, "bj_class"] = "Mage"
                    df.at[i, "bj_subclass"] = "Artillery"
                elif col == 'blueadc':
                    df.at[i, "bb_class"] = "Mage"
                    df.at[i, "bb_subclass"] = "Artillery"
                elif col == 'bluesupport':
                    df.at[i, "bs_class"] = "Mage"
                    df.at[i, "bs_subclass"] = "Artillery"
                elif col == 'redtop':
                    df.at[i, "rt_class"] = "Mage"
                    df.at[i, "rt_subclass"] = "Artillery"
                elif col == 'redmid':
                    df.at[i, "rm_class"] = "Mage"
                    df.at[i, "rm_subclass"] = "Artillery"
                elif col == 'redjungle':
                    df.at[i, "rj_class"] = "Mage"
                    df.at[i, "rj_subclass"] = "Artillery"
                elif col == 'redadc':
                    df.at[i, "rb_class"] = "Mage"
                    df.at[i, "rb_subclass"] = "Artillery"
                elif col == 'redsupport':
                    df.at[i, "rs_class"]= "Mage"
                    df.at[i, "rs_subclass"] = "Artillery"
                    
            elif(df[col][i] == "Anivia" or df[col][i] == "Rumble" or
              df[col][i] == "AurelionSol" or df[col][i] == "Ryze" or
              df[col][i] == "Cassiopeia" or df[col][i] == "Swain" or
              df[col][i] == "Karthus" or df[col][i] == "Taliyah" or
              df[col][i] == "Malzahar" or df[col][i] == "Viktor" or
              df[col][i] == "Vladimir"):
                if col == 'bluetop':
                    df.at[i, "bt_class"] = "Mage"
                    df.at[i, "bt_subclass"] = "Battlemage"
                elif col == 'bluemid':
                    df.at[i, "bm_class"] = "Mage"
                    df.at[i, "bm_subclass"] = "Battlemage"
                elif col == 'bluejungle':
                    df.at[i, "bj_class"] = "Mage"
                    df.at[i, "bj_subclass"] = "Battlemage"
                elif col == 'blueadc':
                    df.at[i, "bb_class"] = "Mage"
                    df.at[i, "bb_subclass"] = "Battlemage"
                elif col == 'bluesupport':
                    df.at[i, "bs_class"] = "Mage"
                    df.at[i, "bs_subclass"] = "Battlemage"
                elif col == 'redtop':
                    df.at[i, "rt_class"] = "Mage"
                    df.at[i, "rt_subclass"] = "Battlemage"
                elif col == 'redmid':
                    df.at[i, "rm_class"] = "Mage"
                    df.at[i, "rm_subclass"] = "Battlemage"
                elif col == 'redjungle':
                    df.at[i, "rj_class"] = "Mage"
                    df.at[i, "rj_subclass"] = "Battlemage"
                elif col == 'redadc':
                    df.at[i, "rb_class"]= "Mage"
                    df.at[i, "rb_subclass"] = "Battlemage"
                elif col == 'redsupport':
                    df.at[i, "rs_class"] = "Mage"
                    df.at[i, "rs_subclass"] = "Battlemage"
                    
            elif(df[col][i] == "Ahri" or df[col][i] == "Lux" or
              df[col][i] == "Annie" or df[col][i] == "Orianna" or
              df[col][i] == "Brand" or df[col][i] == "Sylas" or
              df[col][i] == "Karma" or df[col][i] == "Syndra" or
              df[col][i] == "Leblanc" or df[col][i] == "TwistedFate" or
              df[col][i] == "Lissandra"  or df[col][i] == "Veigar" or
              df[col][i] == "Lux"):
                if col == 'bluetop':
                    df.at[i, "bt_class"] = "Mage"
                    df.at[i, "bt_subclass"] = "Burst"
                elif col == 'bluemid':
                    df.at[i, "bm_class"] = "Mage"
                    df.at[i, "bm_subclass"] = "Burst"
                elif col == 'bluejungle':
                    df.at[i, "bj_class"] = "Mage"
                    df.at[i, "bj_subclass"] = "Burst"
                elif col == 'blueadc':
                    df.at[i, "bb_class"] = "Mage"
                    df.at[i, "bb_subclass"] = "Burst"
                elif col == 'bluesupport':
                    df.at[i, "bs_class"] = "Mage"
                    df.at[i, "bs_subclass"] = "Burst"
                elif col == 'redtop':
                    df.at[i, "rt_class"] = "Mage"
                    df.at[i, "rt_subclass"] = "Burst"
                elif col == 'redmid':
                    df.at[i, "rm_class"] = "Mage"
                    df.at[i, "rm_subclass"] = "Burst"
                elif col == 'redjungle':
                    df.at[i, "rj_class"] = "Mage"
                    df.at[i, "rj_subclass"] = "Burst"
                elif col == 'redadc':
                    df.at[i, "rb_class"]= "Mage"
                    df.at[i, "rb_subclass"] = "Burst"
                elif col == 'redsupport':
                    df.at[i, "rs_class"] = "Mage"
                    df.at[i, "rs_subclass"] = "Burst"
                    
            elif(df[col][i] == "Aphelios" or df[col][i] == "KogMaw" or
              df[col][i] == "Ashe" or df[col][i] == "Lucian" or
              df[col][i] == "Caitlyn" or df[col][i] == "MissFortune" or
              df[col][i] == "Corki" or df[col][i] == "Quinn" or
              df[col][i] == "Draven" or df[col][i] == "Senna" or
              df[col][i] == "Ezreal" or df[col][i] == "Sivir" or
              df[col][i] == "Jhin" or df[col][i] == "Tristana" or
              df[col][i] == "Jinx" or df[col][i] == "Twitch" or
              df[col][i] == "Kaisa" or df[col][i] == "Vayne" or
              df[col][i] == "Kalista" or df[col][i] == "Xayah" or
              df[col][i] == "Kindred" or df[col][i] == "Varus"):
                if col == 'bluetop':
                    df.at[i, "bt_class"] = "Marksman"
                    df.at[i, "bt_subclass"] = "A"
                elif col == 'bluemid':
                    df.at[i, "bm_class"] = "Marksman"
                    df.at[i, "bm_subclass"] = "A"
                elif col == 'bluejungle':
                    df.at[i, "bj_class"] = "Marksman"
                    df.at[i, "bj_subclass"] = "A"
                elif col == 'blueadc':
                    df.at[i, "bb_class"] = "Marksman"
                    df.at[i, "bb_subclass"] = "A"
                elif col == 'bluesupport':
                    df.at[i, "bs_class"] = "Marksman"
                    df.at[i, "bs_subclass"] = "A"
                elif col == 'redtop':
                    df.at[i, "rt_class"] = "Marksman"
                    df.at[i, "rt_subclass"]= "A"
                elif col == 'redmid':
                    df.at[i, "rm_class"] = "Marksman"
                    df.at[i, "rm_subclass"] = "A"
                elif col == 'redjungle':
                    df.at[i, "rj_class"] = "Marksman"
                    df.at[i, "rj_subclass"] = "A"
                elif col == 'redadc':
                    df.at[i, "rb_class"] = "Marksman"
                    df.at[i, "rb_subclass"] = "A"
                elif col == 'redsupport':
                    df.at[i, "rs_class"] = "Marksman"
                    df.at[i, "rs_subclass"] = "A"
                    
            elif(df[col][i] == "Akali" or df[col][i] == "Talon" or
              df[col][i] == "Ekko" or df[col][i] == "Yone" or
              df[col][i] == "Evelynn" or df[col][i] == "Zed" or
              df[col][i] == "Fizz" or df[col][i] == "Leblanc" or
              df[col][i] == "Kassadin" or df[col][i] == "Kayn" or
              df[col][i] == "Katarina" or
              df[col][i] == "Khazix"  or
              df[col][i] == "Nocturne"  or
              df[col][i] == "Pyke"  or
              df[col][i] == "Qiyana"  or
              df[col][i] == "Shaco"):
                if col == 'bluetop':
                    df.at[i, "bt_class"] = "Slayer"
                    df.at[i, "bt_subclass"] = "Assassin"
                elif col == 'bluemid':
                    df.at[i, "bm_class"] = "Slayer"
                    df.at[i, "bm_subclass"] = "Assassin"
                elif col == 'bluejungle':
                    df.at[i, "bj_class"] = "Slayer"
                    df.at[i, "bj_subclass"] = "Assassin"
                elif col == 'blueadc':
                    df.at[i, "bb_class"] = "Slayer"
                    df.at[i, "bb_subclass"] = "Assassin"
                elif col == 'bluesupport':
                    df.at[i, "bs_class"] = "Slayer"
                    df.at[i, "bs_subclass"] = "Assassin"
                elif col == 'redtop':
                    df.at[i, "rt_class"] = "Slayer"
                    df.at[i, "rt_subclass"] = "Assassin"
                elif col == 'redmid':
                    df.at[i, "rm_class"] = "Slayer"
                    df.at[i, "rm_subclass"] = "Assassin"
                elif col == 'redjungle':
                    df.at[i, "rj_class"] = "Slayer"
                    df.at[i, "rj_subclass"] = "Assassin"
                elif col == 'redadc':
                    df.at[i, "rb_class"] = "Slayer"
                    df.at[i, "rb_subclass"] = "Assassin"
                elif col == 'redsupport':
                    df.at[i, "rs_class"] = "Slayer"
                    df.at[i, "rs_subclass"] = "Assassin"
                    
                    
            elif(df[col][i] == "Fiora" or df[col][i] == "Tryndamere" or
              df[col][i] == "Jax" or df[col][i] == "Yasuo" or
              df[col][i] == "Lillia" or 
              df[col][i] == "MasterYi" or
              df[col][i] == "Riven"):
                if col == 'bluetop':
                    df.at[i, "bt_class"] = "Slayer"
                    df.at[i, "bt_subclass"] = "Skirmisher"
                elif col == 'bluemid':
                    df.at[i, "bm_class"] = "Slayer"
                    df.at[i, "bm_subclass"] = "Skirmisher"
                elif col == 'bluejungle':
                    df.at[i, "bj_class"]= "Slayer"
                    df.at[i, "bj_subclass"] = "Skirmisher"
                elif col == 'blueadc':
                    df.at[i, "bb_class"] = "Slayer"
                    df.at[i, "bb_subclass"] = "Skirmisher"
                elif col == 'bluesupport':
                    df.at[i, "bs_class"] = "Slayer"
                    df.at[i, "bs_subclass"] = "Skirmisher"
                elif col == 'redtop':
                    df.at[i, "rt_class"] = "Slayer"
                    df.at[i, "rt_subclass"] = "Skirmisher"
                elif col == 'redmid':
                    df.at[i, "rm_class"] = "Slayer"
                    df.at[i, "rm_subclass"] = "Skirmisher"
                elif col == 'redjungle':
                    df.at[i, "rj_class"]= "Slayer"
                    df.at[i, "rj_subclass"] = "Skirmisher"
                elif col == 'redadc':
                    df.at[i, "rb_class"] = "Slayer"
                    df.at[i, "rb_subclass"]= "Skirmisher"
                elif col == 'redsupport':
                    df.at[i, "rs_class"] = "Slayer"
                    df.at[i, "rs_subclass"] = "Skirmisher"
                    
            elif(df[col][i] == "Alistar" or df[col][i] == "Nautilus" or
              df[col][i] == "Amumu" or df[col][i] == "Sion" or
              df[col][i] == "Gnar" or df[col][i] == "Zac" or
              df[col][i] == "Gragas" or
              df[col][i] == "Leona" or
              df[col][i] == "Malphite" or
              df[col][i] == "Maokai"  or
              df[col][i] == "Nunu"  or
              df[col][i] == "Ornn"  or
              df[col][i] == "Rammus"  or
              df[col][i] == "Sejuani"):
                if col == 'bluetop':
                    df.at[i, "bt_class"] = "Tank"
                    df.at[i, "bt_subclass"] = "Vanguard"
                elif col == 'bluemid':
                    df.at[i, "bm_class"] = "Tank"
                    df.at[i, "bm_subclass"] = "Vanguard"
                elif col == 'bluejungle':
                    df.at[i, "bj_class"] = "Tank"
                    df.at[i, "bj_subclass"]= "Vanguard"
                elif col == 'blueadc':
                    df.at[i, "bb_class"] = "Tank"
                    df.at[i, "bb_subclass"] = "Vanguard"
                elif col == 'bluesupport':
                    df.at[i, "bs_class"] = "Tank"
                    df.at[i, "bs_subclass"] = "Vanguard"
                elif col == 'redtop':
                    df.at[i, "rt_class"] = "Tank"
                    df.at[i, "rt_subclass"] = "Vanguard"
                elif col == 'redmid':
                    df.at[i, "rm_class"] = "Tank"
                    df.at[i, "rm_subclass"] = "Vanguard"
                elif col == 'redjungle':
                    df.at[i, "rj_class"]= "Tank"
                    df.at[i, "rj_subclass"] = "Vanguard"
                elif col == 'redadc':
                    df.at[i, "rb_class"] = "Tank"
                    df.at[i, "rb_subclass"] = "Vanguard"
                elif col == 'redsupport':
                    df.at[i, "rs_class"] = "Tank"
                    df.at[i, "rs_subclass"] = "Vanguard"
                    
            elif(df[col][i] == "Braum" or df[col][i] == "Shen" or
              df[col][i] == "Chogath" or df[col][i] == "TahmKench" or
              df[col][i] == "Galio" or df[col][i] == "Taric" or
              df[col][i] == "Poppy"):
                if col == 'bluetop':
                    df.at[i, "bt_class"] = "Tank"
                    df.at[i, "bt_subclass"] = "Warden"
                elif col == 'bluemid':
                    df.at[i, "bm_class"] = "Tank"
                    df.at[i, "bm_subclass"] = "Warden"
                elif col == 'bluejungle':
                    df.at[i, "bj_class"] = "Tank"
                    df.at[i, "bj_subclass"] = "Warden"
                elif col == 'blueadc':
                    df.at[i, "bb_class"] = "Tank"
                    df.at[i, "bb_subclass"] = "Warden"
                elif col == 'bluesupport':
                    df.at[i, "bs_class"]= "Tank"
                    df.at[i, "bs_subclass"] = "Warden"
                elif col == 'redtop':
                    df.at[i, "rt_class"] = "Tank"
                    df.at[i, "rt_subclass"] = "Warden"
                elif col == 'redmid':
                    df.at[i, "rm_class"] = "Tank"
                    df.at[i, "rm_subclass"]= "Warden"
                elif col == 'redjungle':
                    df.at[i, "rj_class"] = "Tank"
                    df.at[i, "rj_subclass"] = "Warden"
                elif col == 'redadc':
                    df.at[i, "rb_class"] = "Tank"
                    df.at[i, "rb_subclass"] = "Warden"
                elif col == 'redsupport':
                    df.at[i, "rs_class"]= "Tank"
                    df.at[i, "rs_subclass"] = "Warden"
                    
            elif(df[col][i] == "Azir" or
              df[col][i] == "Fiddlesticks" or
              df[col][i] == "Gangplank" or
              df[col][i] == "Graves" or
              df[col][i] == "Heimerdinger" or
              df[col][i] == "Kayle" or
              df[col][i] == "Kennen" or 
              df[col][i] == "Nidalee" or
              df[col][i] == "Singed" or
              df[col][i] == "Teemo" or
              df[col][i] == "Zilean"):
                if col == 'bluetop':
                    df.at[i, "bt_class"] = "Specialist"
                    df.at[i, "bt_subclass"] = "B"
                elif col == 'bluemid':
                    df.at[i, "bm_class"] = "Specialist"
                    df.at[i, "bm_subclass"] = "B"
                elif col == 'bluejungle':
                    df.at[i, "bj_class"] = "Specialist"
                    df.at[i, "bj_subclass"] = "B"
                elif col == 'blueadc':
                    df.at[i, "bb_class"] = "Specialist"
                    df.at[i, "bb_subclass"] = "B"
                elif col == 'bluesupport':
                    df.at[i, "bs_class"] = "Specialist"
                    df.at[i, "bs_subclass"] = "B"
                elif col == 'redtop':
                    df.at[i, "rt_class"] = "Specialist"
                    df.at[i, "rt_subclass"] = "B"
                elif col == 'redmid':
                    df.at[i, "rm_class"] = "Specialist"
                    df.at[i, "rm_subclass"] = "B"
                elif col == 'redjungle':
                    df.at[i, "rj_class"] = "Specialist"
                    df.at[i, "rj_subclass"] = "B"
                elif col == 'redadc':
                    df.at[i, "rb_class"] = "Specialist"
                    df.at[i, "rb_subclass"] = "B"
                elif col == 'redsupport':
                    df.at[i, "rs_class"] = "Specialist"
                    df.at[i, "rs_subclass"] = "B"
                    
                    


# In[25]:


df


# In[26]:


for col in df:
    if col != 'result':
        for i in range(len(df['bluetop'])):
            
            if(i%1000==0):
                print(i)
                
            if(df[col][i] == 'NA'):
                print("s ", col, i, df['bluemid'][i])
                print(col, i, df['bluetop'][i])
                print(col, i, df['bluejungle'][i])
                print(col, i, df['bluesupport'][i])
                print(col, i, df['blueadc'][i])
                print(col, i, df['redtop'][i])
                print(col, i, df['redmid'][i])
                print(col, i, df['redjungle'][i])
                print(col, i, df['redadc'][i])
                print(col, i, df['redsupport'][i], ' e')


# In[27]:


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
            

print(champs_)


# In[28]:


dic


# In[29]:



for col in df:
    if col != 'result':
        for i in range(len(df['bluetop'])):
            
            if(i%1000==0):
                print(i)
                
            #print(df[col][i])
            df.at[i, col] = dic[df[col][i]]
            
df


# In[412]:


X = df.copy()


# ### Randomize SHuffle

# In[413]:


X=X.sample(frac=1)
X


# In[414]:


len(X.columns)


# ### Sklearn

# In[415]:


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

results = X['result']
X = X.drop(axis=1,labels=['result'])
X = X.drop(axis=1,labels=['bluetop','redtop','bluejungle','redjungle','bluemid','redmid','blueadc','redadc','bluesupport','redsupport'])
#X = X.drop(axis=1,labels=["rs_subclass", "rs_class","bs_subclass", "bs_class", "rb_subclass", "rb_class", "bb_subclass", "bb_class", "rj_subclass", "rj_class", "bj_subclass", "bj_class", "rm_subclass", "rm_class", "bm_subclass", "bm_class", "rt_subclass", "rt_class", "bt_subclass", "bt_class"])


# In[411]:


X


# ### Tensorflow

# In[345]:


import tensorflow as tf


# In[353]:


model_emb = tf.keras.models.Sequential([tf.keras.layers.Embedding(168, 10, input_length=30)])

model = tf.keras.models.Sequential([tf.keras.layers.Embedding(170, 10, input_length=10),
                                   tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(200, activation='tanh', 
                                                         kernel_initializer='uniform'),
                                    tf.keras.layers.Dropout(0.3),
                                    tf.keras.layers.Dense(500, activation='tanh', 
                                                         kernel_initializer='uniform'),
                                    tf.keras.layers.Dropout(0.6),
                                    tf.keras.layers.Dense(100, activation='tanh', 
                                                         kernel_initializer='uniform'),
                                    tf.keras.layers.Dropout(0.6),
                                    tf.keras.layers.Dense(500, activation='tanh', 
                                                         kernel_initializer='uniform'),
                                    tf.keras.layers.Dropout(0.6),
                                    tf.keras.layers.Dense(200, activation='tanh', 
                                                         kernel_initializer='uniform'),
                                    tf.keras.layers.Dropout(0.6),
                                    tf.keras.layers.Dense(1, activation='sigmoid')
                                   ])


# In[354]:


model_emb.compile('rmsprop', 'mse')


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-1,
    decay_steps=10000,
    decay_rate=0.9)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[355]:


input_array = np.asarray(X.values).astype(np.float32)
target = np.asarray(results.values)
target


# In[356]:


#output_array = model_emb.predict(input_array)
#print(output_array.shape)
#output_array


# In[379]:


#X_train, X_test, y_train, y_test = train_test_split(output_array, results, random_state=19)
X_train, X_test, y_train, y_test = train_test_split(input_array, target, random_state=3)


# In[374]:


history = model.fit(X_train, y_train, epochs=10, batch_size=1000, validation_split=.30)


# In[375]:


model.evaluate(X_test, y_test)


# In[361]:


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


# ### Naive Bayes

# In[416]:


X = X.reset_index()
X = X.drop(axis=1, labels=['index'])
X


# In[ ]:


for col in X:
    if col != 'result':
        for i in range(len(X['bluetop'])):
            
            if(i%1000==0):
                print(i)
                
            #print(df[col][i])
            X.at[i, col] = str(X[col][i])


# In[419]:


input_array = np.asarray(X.values).astype(np.float32)
target = np.asarray(results.values)
target


# In[420]:


X_train, X_test, y_train, y_test = train_test_split(input_array, target, random_state=3)


# In[423]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(np.array(X_train).astype(int), np.array(y_train).astype(int))
np.array(X_train).astype(int)


# In[425]:


clf.score(np.array(X_test).astype(int), np.array(y_test).astype(int))


# In[ ]:




