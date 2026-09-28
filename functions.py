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

def grafico_boxplots(df, colunas=COLUNAS_NUMERICAS, nome_arquivo='boxplots.png'):
    fig, eixos = plt.subplots(2, (len(colunas) + 1) // 2, figsize=(14, 8))
    for ax, c in zip(eixos.flat, colunas):
        sns.boxplot(y=df[c], ax=ax, color='#4c72b0')
        ax.set_title(c)
        ax.set_ylabel('')
    for ax in list(eixos.flat)[len(colunas):]:
        ax.set_visible(False)
    fig.suptitle('Boxplots das variáveis numéricas')
    return _salvar(fig, nome_arquivo)

def grafico_histograma(serie, nome_arquivo=None, bins=30):
    serie = serie.dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(serie, bins=bins, stat='density', kde=True, ax=ax, color='#4c72b0', label='Dados')
    x = np.linspace(serie.min(), serie.max(), 300)
    ax.plot(x, stats.norm.pdf(x, serie.mean(), serie.std()), 'r--', label='Normal teórica')
    ax.axvline(serie.mean(), color='green', linestyle=':', label=f'Média = {serie.mean():.2f}')
    ax.axvline(serie.median(), color='orange', linestyle=':', label=f'Mediana = {serie.median():.2f}')
    ax.set_title(f'Distribuição de {serie.name}')
    ax.legend()
    return _salvar(fig, nome_arquivo or f'histograma_{serie.name}.png')

def grafico_heatmap(corr, titulo_grafico='Matriz de correlação', nome_arquivo='heatmap_correlacao.png'):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title(titulo_grafico)
    return _salvar(fig, nome_arquivo)

def grafico_barras(serie, titulo_grafico, nome_arquivo, xlabel='', ylabel='', horizontal=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    serie.plot(kind='barh' if horizontal else 'bar', ax=ax, color='#4c72b0')
    ax.set_title(titulo_grafico)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _salvar(fig, nome_arquivo)

def grafico_linha(serie, titulo_grafico, nome_arquivo, xlabel='', ylabel=''):
    fig, ax = plt.subplots(figsize=(11, 5))
    serie.plot(ax=ax, marker='o')
    ax.set_title(titulo_grafico)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return _salvar(fig, nome_arquivo)

def vendas_por(df, coluna, valor='Total_Venda', agregacao='sum'):
    return df.groupby(coluna, observed=True)[valor].agg(agregacao).sort_values(ascending=False)

def vendas_diarias(df, valor='Total_Venda'):
    return df.set_index('Data')[valor].resample('D').sum()

def filtrar_pedidos(df, inicio, fim):
    return df[(df['Data'] >= pd.Timestamp(inicio)) & (df['Data'] <= pd.Timestamp(fim))]

def semana_em_torno(data_central, dias_antes=3, dias_depois=3):
    centro = pd.Timestamp(data_central)
    return centro - pd.Timedelta(days=dias_antes), centro + pd.Timedelta(days=dias_depois)

def coeficiente_variacao_mensal(df, grupo='Produto', valor='Total_Venda'):
    mensal = df.pivot_table(index=grupo, columns='Mes', values=valor, aggfunc='sum', fill_value=0)
    cv = mensal.std(axis=1) / mensal.mean(axis=1) * 100
    return cv.sort_values(ascending=False).rename('cv_mensal(%)')
