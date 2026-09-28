from pathlib import Path
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# CONSTANTES
COLUNAS_NUMERICAS = ['Preço', 'Quantidade', 'Desconto', 'Total_Venda', 'Idade', 'Renda']
DIAS_SEMANA = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
MESES = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
FAIXAS_ETARIAS = ['Jovem', 'Adulto', 'Idoso']
ALPHA = 0.05
PASTA_GRAFICOS = Path(__file__).parent / 'graficos'
PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

def carregar_dados(caminho):
    df = pd.read_csv(caminho)
    df['Data'] = pd.to_datetime(df['Data'])
    df['Mes'] = df['Data'].dt.month
    df['Nome_Mes'] = pd.Categorical(df['Mes'].map(lambda m: MESES[m - 1]), categories=MESES, ordered=True)
    df['Dia_Semana'] = pd.Categorical(df['Data'].dt.dayofweek.map(lambda d: DIAS_SEMANA[d]), categories=DIAS_SEMANA, ordered=True)
    df['Faixa_Etaria'] = classificar_faixa_etaria(df['Idade'])
    df['Tem_Desconto'] = df['Desconto'] > 0
    return df
