import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
#as duas linhas abaixo são necessarias para o plotly rodar no spyder.
import plotly.io as pio
pio.renderers.default='browser'




base_census = pd.read_csv('census.csv')

base_census.describe()

base_census.isnull().sum()

np.unique(base_census['income'], return_counts=True)

sns.countplot(x = base_census['income'])

plt.hist(x = base_census['age'])

plt.hist(x = base_census['education-num'])

plt.hist(x = base_census['hour-per-week'])

# Grafico interessante para agrupar dados
grafico = px.treemap(base_census, 
                     path=['occupation','relationship','age'],)

grafico.show()

grafico2 = px.parallel_categories(base_census,dimensions=['education','income'],)

grafico2.show()

X_census = base_census.iloc[:, 0:14].values

y_census = base_census.iloc[:, 14].values

#Tratamento de atributos categóricos

#Label Encoder

from sklearn.preprocessing import LabelEncoder

label_encoder_teste = LabelEncoder()
teste = label_encoder_teste.fit_transform(X_census[:,1])

np.unique(teste,return_counts=True)

label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital_status = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

X_census[:,1] = label_encoder_workclass.fit_transform(X_census[:,1])
X_census[:,3] = label_encoder_education.fit_transform(X_census[:,3])
X_census[:,5] = label_encoder_marital_status.fit_transform(X_census[:,5])
X_census[:,6] = label_encoder_occupation.fit_transform(X_census[:,6])
X_census[:,7] = label_encoder_relationship.fit_transform(X_census[:,7])
X_census[:,8] = label_encoder_race.fit_transform(X_census[:,8])
X_census[:,9] = label_encoder_sex.fit_transform(X_census[:,9])
X_census[:,13] = label_encoder_country.fit_transform(X_census[:,13])

#label enconder tem problema devido ao algoritmo acabar dando maior importância aos valores maiores


#OneHotEncoder
# Corrige o problema do label encoder
# Porém adicionar mais colunas (atributos) no dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(),
                                                        [1,3,5,6,7,8,9,13])],
                                         remainder='passthrough') 
# o remainder = 'passthrough' é para não apagar os demais atributos que não são categóricos

X_census = onehotencoder_census.fit_transform(X_census).toarray()

X_census.shape

from sklearn.preprocessing import StandardScaler

scaler_census = StandardScaler()
X_census_scaled = scaler_census.fit_transform(X_census)


# Avaliação do algoritmo.

from sklearn.model_selection import train_test_split

RANDOM_STATE = 0

X_census_train, X_census_test, y_census_train, y_census_test = train_test_split(
    X_census_scaled,y_census, test_size=0.15, random_state= RANDOM_STATE)

#salvando as bases de dados

import pickle


with open('census.pkl',mode = 'wb') as f:
    pickle.dump([X_census_train,y_census_train,X_census_test,y_census_test], f)
    
    
# lendo o arquivo pkl

with open('census.pkl',mode = 'rb') as f:
    X_census_train,y_census_train,X_census_test,y_census_test = pickle.load(f)
    
#aplicando o algoritmo de classificação
from sklearn.naive_bayes import GaussianNB

naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_census_train, y_census_train)

previsao = naive_bayes_classifier.predict(X_census_test)

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

accuracy_score(y_census_test, previsao)
confusion_matrix(y_census_test, previsao)

from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(naive_bayes_classifier)
cm.fit(X_census_train,y_census_train)
cm.score(X_census_test,y_census_test)

report = classification_report(y_census_test, previsao)
