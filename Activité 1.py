#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


# # Import et préparation du Dataset

# In[2]:


data = pd.read_csv("les-arbres.csv", sep=";")
data.head()


# In[3]:


print(any(pd.isnull(data["CIRCONFERENCEENCM"])))
print(any(pd.isna(data["CIRCONFERENCEENCM"])))
print(any(pd.isnull(data["HAUTEUR (m)"])))
print(any(pd.isna(data["HAUTEUR (m)"])))


# In[4]:


data[data["HAUTEUR (m)"] > 80]


# <p> Au vu de la circonférence des arbres de + de 80m, il semble qu'il y ait eu une erreur quant à l'unité utilisé ; <br/>
#     La hauteur de ces arbres est exprimée en cm et non en mètre, on pourrait changer les unités, mais je vais me contenter d'enlever les valeurs.</p>

# In[5]:


data = data.loc[data['HAUTEUR (m)'] < 80]


# <p> On change le nom des colonnes Hauteur et circonférence pour les avoir en attachées (utile pour après)

# In[6]:


data = data.rename(columns={'HAUTEUR (m)':"haut"})
data = data.rename(columns={'CIRCONFERENCEENCM':"circ"})


# # Affichage du nuage de points

# In[12]:


sns.set()

ax = sns.scatterplot(x="haut", y="circ", data=data)
ax.set(xlabel='Hauteur (m)', ylabel="Circonférence (en cm)")
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
fig = ax.get_figure()
fig.savefig("Graphiques/nuage_de_points.png")


# <p> Il semble bien que l'on ait une relation linéaire sur ce nuage de points ; <br/>
#     Vérifions par le calcul : </p>

# # Régression linéaire 

# In[13]:


reg_simp = smf.ols('circ ~ haut', data=data).fit()


# In[14]:


print(reg_simp.summary())


# In[15]:


r_squared = reg_simp.rsquared
print("Le coéfficient de Détermination est de : ", r_squared.round(2))
print('\n')
p_value = reg_simp.pvalues[1]
print("La p_valeur est égale à :", p_value)


# <p> Le nuage originel est assez dispersé (surtout dans les grandes valeurs de circonférence), cela explique que notre coéfficient de détermination soit seulement à 0,62 </p>
# <p> La p_valeur est inférieur à Alpha (niveau de test de 5%), on rejette donc l'hypothèse selon laquelle le paramètre vaut 0 ; On voit ici que la variable circonférence est significative. <p>

# In[ ]:




