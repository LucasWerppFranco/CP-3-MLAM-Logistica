import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from datetime import datetime, timedelta

# Função para carregar ou simular dados históricos de vendas
def carregar_dados():
    # Simulando dados de 24 meses para um produto
    datas = pd.date_range(start='2022-01-01', periods=24, freq='M')
    np.random.seed(42)
    vendas_base = np.array([100, 120, 110, 130, 140, 150, 160, 170, 180, 190, 200, 210,
                            220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
    # Adicionando ruído aleatório para simular variabilidade real
    vendas = vendas_base + np.random.normal(0, 15, 24)
    df = pd.DataFrame({'Data': datas, 'Vendas': vendas})
    df.set_index('Data', inplace=True)
    return df

# Função para análise estatística descritiva
def analise_descritiva(df):
    media = df['Vendas'].mean()
    mediana = df['Vendas'].median()
    desvio_padrao = df['Vendas'].std()
    variancia = df['Vendas'].var()
    
    print("=== ANÁLISE ESTATÍSTICA DESCRITIVA ===")
    print(f"Média das vendas mensais: {media:.2f} unidades")
    print(f"Mediana das vendas mensais: {mediana:.2f} unidades")
    print(f"Desvio padrão: {desvio_padrao:.2f} (indica variabilidade moderada)")
    print(f"Variação: {variancia:.2f}")
    print("\nInterpretação: A demanda está crescendo, mas com flutuações. Recomenda-se buffer de estoque baseado no desvio padrão.")

# Função para calcular média móvel (suavização de série temporal)
def media_movel(df, janela=3):
    df['Media_Movel'] = df['Vendas'].rolling(window=janela).mean()
    return df

# Função para regressão linear e previsão de demanda
def prever_demanda(df):
    # Preparar dados para regressão: X = meses (numéricos), y = vendas
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Vendas'].values
    
    # Regressão linear simples
    slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
    
    # Modelo: y = slope * x + intercept
    print("\n=== REGRESSÃO LINEAR ===")
    print(f"Equação: Vendas = {slope:.2f} * Mês + {intercept:.2f}")
    print(f"Coeficiente de determinação (R²): {r_value**2:.4f} (boa adequação do modelo)")
    print(f"P-valor: {p_value:.4f} (significativo, p < 0.05)")
    
    # Previsão para os próximos 3 meses
    proximos_meses = np.array([len(df), len(df)+1, len(df)+2]).reshape(-1, 1)
    X_com_const = sm.add_constant(proximos_meses)  # Para statsmodels
    modelo = sm.OLS(y, sm.add_constant(X)).fit()
    previsoes = modelo.predict(X_com_const)
    
    # Adicionar buffer de 20% para otimização (considerando desvio padrão)
    buffer = 0.20
    previsoes_otimizadas = previsoes * (1 + buffer)
    
    print("\nPrevisões para os próximos 3 meses (com buffer de 20%):")
    for i, pred in enumerate(previsoes_otimizadas, 1):
        print(f"Mês {len(df) + i}: {pred:.2f} unidades (previsão base: {previsoes[i-1]:.2f})")
    
    # Recomendação de otimização
    total_recomendado = sum(previsoes_otimizadas)
    print(f"\nRecomendação de Logística: Peça {total_recomendado:.0f} unidades no total para os próximos 3 meses.")
    print("Isso otimiza o estoque, evitando faltas (stockout) e excessos.")
    
    return previsoes, previsoes_otimizadas, slope, intercept

# Função para visualização
def plotar_graficos(df, previsoes, slope, intercept):
    plt.figure(figsize=(12, 8))
    
    # Gráfico 1: Vendas históricas e média móvel
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Vendas'], label='Vendas Reais', marker='o')
    plt.plot(df.index, df['Media_Movel'], label='Média Móvel (janela=3)', linestyle='--')
    plt.title('Análise de Demanda Histórica - Logística de Estoque')
    plt.xlabel('Data')
    plt.ylabel('Unidades Vendidas')
    plt.legend()
    plt.grid(True)
    
    # Gráfico 2: Regressão linear e previsões
    plt.subplot(2, 1, 2)
    X_hist = np.arange(len(df))
    plt.plot(X_hist, df['Vendas'], label='Vendas Históricas', marker='o')
    plt.plot(X_hist, slope * X_hist + intercept, label='Linha de Regressão', color='red')
    
    # Previsões futuras
    X_fut = np.array([len(df), len(df)+1, len(df)+2])
    plt.plot(X_fut, previsoes, label='Previsão Base', marker='s', color='green')
    plt.plot(X_fut, previsoes * 1.20, label='Previsão Otimizada (com buffer)', marker='s', color='orange')
    
    plt.title('Previsão de Demanda e Otimização de Estoque')
    plt.xlabel('Mês (0 = Jan/2022)')
    plt.ylabel('Unidades Vendidas')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Função principal para executar a aplicação
def main():
    print("=== APLICAÇÃO DE OTIMIZAÇÃO DE ESTOQUE EM LOGÍSTICA ===")
    print("Contexto: Previsão de demanda para smartphones em um armazém de e-commerce.\n")
    
    # Carregar dados
    df = carregar_dados()
    
    # Análise descritiva
    analise_descritiva(df)
    
    # Média móvel
    df = media_movel(df)
    
    # Previsão
    previsoes_base, previsoes_otim, slope, intercept = prever_demanda(df)
    
    # Visualização
    plotar_graficos(df, previsoes_base, slope, intercept)
    
    # Salvar relatório em CSV para o gerente
    df.to_csv('relatorio_estoque.csv')
    print("\nRelatório salvo em 'relatorio_estoque.csv' para análise adicional.")

# Executar a aplicação
if __name__ == "__main__":
    main()

