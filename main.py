# 1. Quais foram as categorias produtos mais vendidos e em quais meses eles tiveram o maior pico de vendas?
# 2. Qual foi o impacto das promoções (descontos aplicados) nas vendas? Quais produtos tiveram o maior aumento nas vendas durante as promoções? 
# 3. Qual a média de vendas diárias por categoria de produto?
# 4. Quais produtos apresentam a maior sazonalidade nas vendas (ex.: frutas e vegetais, sorvetes)? 
#   a. Para responder essa pergunta considere a medida de sazonalidade dada pelo coeficiente de variação mensal dos produtos.
# 5. Quais as faixas etárias contribuíram mais para as vendas totais? 
#   a. Para responder essa pergunta, crie uma coluna com a faixa etária, considerando: entre 18 e 35 (jovem), entre 36 e 59 (adulto) e acima de 60 anos (idoso). 
# 6. Qual a distribuição das vendas ao longo dos dias da semana?
# 7. Como as vendas mensais se comparam antes e durante a Black Friday?
# 8. Qual é o ticket médio (valor médio das vendas) por faixa etária e como ele varia entre diferentes categorias de produto? 
# 9. Qual é a relação entre a idade dos clientes e o valor total das compras? 
# 10. Comparando a semana da "Black Friday" com a do Natal, qual semana a empresa teve melhores desempenhos em relação à média de vendas? 

# === CONTENT ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import functions as f

pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.width = 200
pd.options.display.max_columns = None

# Carrega dados do arquivo CSV e cria colunas adicionais para análise
CSV_PATH = Path(__file__).parent / 'csv' / 'vendas_supermercado_com_cliente.csv'
data = f.carregar_dados(CSV_PATH)
print(data.head())

# 1. Quais foram as categorias produtos mais vendidos e em quais meses eles tiveram o maior pico de vendas?
mais_vendidos = f.vendas_por(data, 'Categoria')
vendas_mensais_cat = data.pivot_table(index='Categoria', columns='Nome_Mes', values='Total_Venda',
                                      aggfunc='sum', observed=True)
p1 = pd.DataFrame({
    'Total (R$)': mais_vendidos,
    'Part. (%)': mais_vendidos / mais_vendidos.sum() * 100,
    'Mês de pico': vendas_mensais_cat.idxmax(axis=1),
    'Vendas no pico (R$)': vendas_mensais_cat.max(axis=1),
}).loc[mais_vendidos.index]

print('\n1. Categorias mais vendidas (R$) e mês de pico:')
print(p1)
f.grafico_barras(mais_vendidos, 'Total vendido por categoria em 2023', 'p1_categorias.png',
                 xlabel='Categoria', ylabel='Total vendido (R$)')
print(f'Interpretação: {mais_vendidos.index[0]} lidera com R$ {mais_vendidos.iloc[0]:,.2f} '
      f'({p1["Part. (%)"].iloc[0]:.1f}% do faturamento), '
      f'{mais_vendidos.iloc[0] / mais_vendidos.iloc[1]:.1f}x a 2ª colocada ({mais_vendidos.index[1]}). '
      f'Os meses de pico mostram quando cada categoria precisa de mais estoque e reforço de promoções.')

# 2. Qual foi o impacto das promoções (descontos aplicados) nas vendas? Quais produtos tiveram o maior aumento nas vendas durante as promoções? 
impacto_geral = data.groupby('Tem_Desconto')[['Total_Venda', 'Quantidade']].mean()
qtd_produto = data.pivot_table(index='Produto', columns='Tem_Desconto', values='Quantidade', aggfunc='mean')
produto_maior_aumento = ((qtd_produto[True] / qtd_produto[False] - 1) * 100).sort_values(ascending=False).head(10).rename('Aumento qtd. (%)')
f.grafico_barras(impacto_geral['Total_Venda'], 'Impacto das promoções nas vendas', 'p2_impacto_promocoes.png', xlabel='Tem desconto?', ylabel='Média de vendas (R$)')
print('\n2. Impacto das promoções nas vendas:')
print(impacto_geral)
print('\nProdutos com maior aumento nas vendas durante promoções:')
print(produto_maior_aumento)

# 3. Qual a média de vendas diárias por categoria de produto?
media_vendas_diarias = data.pivot_table(index='Data', columns='Categoria', values='Total_Venda', aggfunc='sum', fill_value=0).mean().sort_values(ascending=False)
print('\n3. Média de vendas diárias por categoria de produto:')
print(media_vendas_diarias)

# 4. Quais produtos apresentam a maior sazonalidade nas vendas (ex.: frutas e vegetais, sorvetes)? 
maior_sazonalidade = f.coeficiente_variacao_mensal(data).head(10)
print('\n4. Produtos com maior sazonalidade nas vendas (coeficiente de variação mensal):')
print(maior_sazonalidade)

# 5. Quais as faixas etárias contribuíram mais para as vendas totais? 
faixa_etaria = f.vendas_por(data, 'Faixa_Etaria')
percentual = faixa_etaria / faixa_etaria.sum() * 100
print('\n5. Contribuição das faixas etárias para as vendas totais:')
print(pd.DataFrame({'Total (R$)': faixa_etaria, 'Part. (%)': percentual}))

# 6. Qual a distribuição das vendas ao longo dos dias da semana?
vd = f.vendas_diarias(data)
grouped_vd = vd.groupby(vd.index.dayofweek).agg(['mean', 'sum', 'count']).set_axis(f.DIAS_SEMANA)
print('\n6. Distribuição das vendas ao longo dos dias da semana:')
print(grouped_vd)

# 7. Como as vendas mensais se comparam antes e durante a Black Friday?
vendas_mensais = f.vendas_por(data, 'Nome_Mes').sort_index()
comparacao_grafico = f.grafico_linha(vendas_mensais, 'Vendas mensais', 'p7_vendas_mensais.png', xlabel='Mês', ylabel='Total vendido (R$)')
print('\n7. Comparação das vendas mensais antes e durante a Black Friday:')
print(vendas_mensais)
print(f'Interpretação: Novembro (Black Friday) vendeu R$ {vendas_mensais["Nov"]:,.2f}, '
      f'{(vendas_mensais["Nov"] / vendas_mensais["Out"] - 1) * 100:.1f}% em relação a outubro e '
      f'{(vendas_mensais["Nov"] / vendas_mensais["Jan":"Out"].mean() - 1) * 100:.1f}% em relação à média de Jan-Out; '
      f'a Black Friday não gerou pico de vendas no mês.')

# 8. Qual é o ticket médio (valor médio das vendas) por faixa etária e como ele varia entre diferentes categorias de produto?
ticket_medio_faixa = f.vendas_por(data, 'Faixa_Etaria', agregacao='mean')
ticket_categoria_faixa = data.pivot_table(index='Categoria', columns='Faixa_Etaria', values='Total_Venda', aggfunc='mean', observed=True)
f.grafico_heatmap(ticket_categoria_faixa, 'Ticket médio (R$) por categoria e faixa etária', 'p8_ticket_categoria_faixa.png')
print('\n8. Ticket médio por faixa etária:')
print(ticket_medio_faixa)
print('\nTicket médio por categoria e faixa etária:')
print(ticket_categoria_faixa)

# 9. Qual é a relação entre a idade dos clientes e o valor total das compras?
r_pearson = data['Idade'].corr(data['Total_Venda'])
r_spearman = data['Idade'].corr(data['Total_Venda'], method='spearman')
print('\n9. Relação entre idade e valor da compra:')
print(f'Pearson: {f.interpretar_correlacao(r_pearson)} | Spearman: {f.interpretar_correlacao(r_spearman)}')

# 10. Comparando a semana da "Black Friday" com a do Natal, qual semana a empresa teve melhores desempenhos em relação à média de vendas?
semana_bf = vd[slice(*f.semana_em_torno('2023-11-24'))]
semana_natal = vd[slice(*f.semana_em_torno('2023-12-25'))]
comparacao_semanas = pd.DataFrame({'Black Friday': semana_bf.agg(['mean', 'sum']), 'Natal': semana_natal.agg(['mean', 'sum'])})
print('\n10. Vendas diárias: semana da Black Friday x semana do Natal:')
print(comparacao_semanas)
