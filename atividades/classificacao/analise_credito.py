import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
#as duas linhas abaixo são necessarias para o plotly rodar no spyder.
import plotly.io as pio
pio.renderers.default='browser'


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
base_credit.loc[base_credit['age'] < 0]

# Apagar a coluna inteira (de todos os registros da base de dados)
base_credit2 = base_credit.drop('age', axis = 1)
print(base_credit2)
base_credit.index

# Apagar somente os registros com valores inconsistentes
base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
base_credit3
base_credit3.loc[base_credit3['age'] < 0]

# Preencher os valores inconsistente manualmente
# Prencher a média
base_credit['age'][base_credit['age'] > 0].mean()
base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92

base_credit.head(27)


#2 - Valores Faltantes
base_credit.isnull()
base_credit.isnull().sum()
base_credit.loc[pd.isnull(base_credit['age'])]
#preenchendo com a média os valores NULL
base_credit['age'].fillna(base_credit['age'].mean(), inplace = True)

base_credit.loc[base_credit['clientid'].isin([29,31,32])]
base_credit.isnull().sum()


# Divisão entre previsores e classe

X_credit = base_credit.iloc[:, 1:4].values

X_credit

y_credit = base_credit.iloc[:, 4].values

y_credit

#Escalonamento dos dados

from sklearn.preprocessing import StandardScaler

scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)

X_credit

from sklearn.model_selection import train_test_split
RANDOM_STATE = 0

X_credit_train,X_credit_test, y_credit_train, y_credit_test = train_test_split(
    X_credit, y_credit, test_size=0.25, random_state=RANDOM_STATE)

#salvando a base de dados

import pickle

with open('credit.pkl',mode = 'wb') as f:
    pickle.dump([X_credit_train,y_credit_train,X_credit_test,y_credit_test], f)
    

# lendo o arquivo pkl

with open('credit.pkl',mode = 'rb') as f:
    X_credit_train,y_credit_train,X_credit_test,y_credit_test = pickle.load(f)
    
#aplicando o algoritmo de classificação
from sklearn.naive_bayes import GaussianNB

naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_credit_train, y_credit_train)

previsao = naive_bayes_classifier.predict(X_credit_test)

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

accuracy_score(y_credit_test, previsao)
confusion_matrix(y_credit_test, previsao)

from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(naive_bayes_classifier)
cm.fit(X_credit_train,y_credit_train)
cm.score(X_credit_test,y_credit_test)

report = classification_report(y_credit_test, previsao)
