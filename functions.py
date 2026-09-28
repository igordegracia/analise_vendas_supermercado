from pathlib import Path
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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

def classificar_faixa_etaria(idades):
    return pd.cut(idades, bins=[0, 35, 59, np.inf], labels=FAIXAS_ETARIAS, ordered=True)

def medidas_descritivas(serie):
    serie = serie.dropna()
    media = serie.mean()
    q1, q2, q3 = serie.quantile([0.25, 0.5, 0.75])
    return pd.Series({
        'media': media,
        'mediana': serie.median(), 
        'moda': serie.mode().iloc[0],
        'desvio_padrao': serie.std(),
        'variancia': serie.var(),
        'minimo': serie.min(),
        'maximo': serie.max(),
        'amplitude': serie.max() - serie.min(),
        'cv(%)': serie.std() / media * 100 if media != 0 else np.nan,
        'p10': serie.quantile(0.1),
        'q1(p25)': q1,
        'q2(p50)': q2,
        'q3(p75)': q3,
        'p90': serie.quantile(0.9),
        'iqr': q3 - q1,
        'assimetria': serie.skew(),
        'curtose': serie.kurt(),
    }, name=serie.name)
