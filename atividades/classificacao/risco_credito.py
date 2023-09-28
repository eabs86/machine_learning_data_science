import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
#as duas linhas abaixo são necessarias para o plotly rodar no spyder.
import plotly.io as pio
pio.renderers.default='browser'

base_risco_credito = pd.read_csv('risco_credito.csv')

X_risco_credito = base_risco_credito.iloc[:,0:4].values

y_risco_credito = base_risco_credito.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder

label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantias = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credito[:,0] = label_encoder_historia.fit_transform(X_risco_credito[:,0])
X_risco_credito[:,1] = label_encoder_divida.fit_transform(X_risco_credito[:,1])
X_risco_credito[:,2] = label_encoder_garantias.fit_transform(X_risco_credito[:,2])
X_risco_credito[:,3] = label_encoder_renda.fit_transform(X_risco_credito[:,3])

#essa base só precisará do LabelEncoder devido ao seu tamnho

import pickle

with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([X_risco_credito,y_risco_credito], f)

from sklearn.naive_bayes import GaussianNB

naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_risco_credito, y_risco_credito)

# teste com registro:
# Primeiro: História boa (0), divida alta (0), garantias nenhuma(1), renda >35 (2)
# Segundo:  História ruim (2), divida alta (0), garantias adequada(0), renda <15 (0)
previsao = naive_bayes_classifier.predict([[0,0,1,2],[2,0,0,0]])

naive_bayes_classifier.classes_
naive_bayes_classifier.class_count_
naive_bayes_classifier.class_prior_