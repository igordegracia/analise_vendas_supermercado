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

def tabela_descritiva(df, colunas=COLUNAS_NUMERICAS):
    return pd.concat([medidas_descritivas(df[coluna]) for coluna in colunas], axis=1)

def interpretar_cv(cv):
    if cv < 15:
        return 'baixa disperção'
    elif cv < 30:
        return 'disperção moderada'
    else:
        return 'alta disperção'

def interpretar_assimetria(s):
    if abs(s) < 0.5:
        return f'assimetria de {s:.2f}: distribuição aproximadamente simétrica'
    if s > 0:
        return (f'assimetria de {s:.2f}: assimetria positiva, poucos valores altos ', f'puxam a média para baixo da mediana.')

def interpretar_curtose(k):
    if abs(k) < 0.5:
        return f'curtose de {k:.2f} → distribuição aproximadamente normal'
    if k > 0:
        return f'curtose de {k:.2f} → distribuição leptocúrtica, com caudas mais pesadas que a normal'
    return f'curtose de { k:.2f} -> distribuição platicúrtica, com caudas mais leves que a normal'

def outliers_iqr(serie, k=1.5):
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - k * iqr, q3 + k * iqr
    return (serie < lim_inf) | (serie > lim_sup), lim_inf, lim_sup

def outliers_zscore(serie, limite=3):
    z = (serie - serie.mean()) / serie.std()
    return z.abs() > limite

def resumo_outliers(df, colunas=COLUNAS_NUMERICAS):
    linhas = []
    for c in colunas:
        mask_iqr, lim_inf, lim_sup = outliers_iqr(df[c])
        mask_z = outliers_zscore(df[c])
        linhas.append({
            'Variável': c,
            'Lim. inferior (IQR)': lim_inf,
            'Lim. superior (IQR)': lim_sup,
            'Outliers IQR': int(mask_iqr.sum()),
            '% IQR': mask_iqr.mean() * 100,
            'Outliers z>3': int(mask_z.sum()),
            '% z>3': mask_z.mean() * 100,
        })
    return pd.DataFrame(linhas).set_index('Variável')

def tabela_frequencia(serie, ordenar_por_frequencia=False):
    contagem = serie.value_counts(sort=ordenar_por_frequencia)
    if not ordenar_por_frequencia and not isinstance(serie.dtype, pd.CategoricalDtype):
        contagem = contagem.sort_index()
    tabela = pd.DataFrame({
        'Freq. Absoluta': contagem,
        'Freq. Relativa (%)': contagem / contagem.sum() * 100,
    })
    tabela['Freq. Acumulada (%)'] = tabela['Freq. Relativa (%)'].cumsum()
    tabela.loc['Total'] = [contagem.sum(), 100.0, np.nan]
    return tabela

def matriz_correlacao(df, colunas=COLUNAS_NUMERICAS, metodo='pearson'):
    return df[colunas].corr(method=metodo)

def interpretar_correlacao(r):
    forca = abs(r)
    if forca < 0.1:
        intensidade = 'desprezível'
    elif forca < 0.3:
        intensidade = 'fraca'
    elif forca < 0.7:
        intensidade = 'moderada'
    else:
        intensidade = 'forte'
    sentido = 'positiva' if r > 0 else 'negativa'
    return f'correlação {intensidade} {sentido} (r = {r:.2f})'

def correlacoes_relevantes(corr, limite=0.1):
    pares = []
    colunas = corr.columns
    for i in range(len(colunas)):
        for j in range(i + 1, len(colunas)):
            r = corr.iloc[i, j]
            if abs(r) >= limite:
                pares.append((colunas[i], colunas[j], r))
    return sorted(pares, key=lambda p: abs(p[2]), reverse=True)

def _salvar(fig, nome_arquivo):
    PASTA_GRAFICOS.mkdir(exist_ok=True)
    caminho = PASTA_GRAFICOS / nome_arquivo
    fig.tight_layout()
    fig.savefig(caminho, dpi=300)
    plt.close(fig)
    print(f'Gráfico salvo em: {caminho}')
    return caminho
