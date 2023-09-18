import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


base_credit = pd.read_csv('credit_data.csv')

#Exploração de Dados
base_credit.head(10)

base_credit.tail(8)

base_credit.describe()

base_credit[base_credit['income'] >= 69995.685578]

base_credit[base_credit['loan'] <= 1.377630]

#Visualização dos Dados

np.unique(base_credit['default'], return_counts=True)

sns.countplot(x = base_credit['default'])

plt.hist(x = base_credit['age'])

plt.hist(x = base_credit['income'])

plt.hist(x = base_credit['loan'])

grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color = 'default')
grafico.show()

#Tratamento dos dados

#1 - Valores Inconsistentes


#2 - Valores Faltantes
