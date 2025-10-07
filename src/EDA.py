"""
Funções para Análise Exploratória de Dados (EDA)
"""
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
import numpy as np

def boxplot(df: pd.DataFrame):
    # Seleciona apenas colunas numéricas
    cols_numericas = df.select_dtypes(include='number').columns.tolist()
    tamanho = len(cols_numericas)

    if tamanho == 0:
        print("O DataFrame não possui colunas numéricas.")
        return

    # Calcula número de linhas (3 colunas por linha)
    linhas = math.ceil(tamanho / 3)
    fig, axes = plt.subplots(linhas, 3, figsize=(14, 4 * linhas))
    axes = axes.flatten()  # transforma em uma lista unidimensional

    # Cria um boxplot para cada coluna numérica
    for i, col in enumerate(cols_numericas):
        sns.boxplot(data=df, y=col, ax=axes[i], color='tomato')
        axes[i].set_title(col)

    # Remove subplots vazios (se houver)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def histograma(df, cols=None, dx=5):
    # Seleciona colunas numéricas automaticamente se não forem especificadas
    if cols is None:
        cols = df.select_dtypes(include='number').columns.tolist()

    tamanho = len(cols)
    if tamanho == 0:
        print("O DataFrame não possui colunas numéricas para plotar.")
        return

    linhas = math.ceil(tamanho / 3)
    fig, axes = plt.subplots(linhas, 3, figsize=(15, 5 * linhas))
    axes = np.array(axes).flatten()

    N = len(df)
    cores = ['tomato', 'teal', 'orange']

    for i, col in enumerate(cols):
        temp = df[col].dropna()
        cor = cores[i % len(cores)]

        minimo, maximo = temp.min(), temp.max()

        # --- Cálculo automático do bin (Freedman–Diaconis) ---
        if dx is None:
            Q1, Q3 = np.percentile(temp, [25, 75])
            IQR = Q3 - Q1
            n = len(temp)
            bin_width = 2 * IQR / (n ** (1/3))
            if bin_width == 0 or np.isnan(bin_width):
                bin_width = (maximo - minimo) / 10  # fallback
            dx_local = bin_width
        else:
            dx_local = dx

        bins = np.arange(minimo, maximo + dx_local, dx_local)
        c, x = np.histogram(temp, bins)
        p = c / N * 100

        axes[i].bar(
            x[0:-1], p,
            width=np.diff(bins),
            align='edge',
            color=cor,
            edgecolor='black'
        )
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Percentual (%)')
        axes[i].set_xticks(bins)

        # Adiciona porcentagens sobre as barras
        for j in range(len(c)):
            axes[i].text(
                x[j] + dx_local / 2,
                p[j],
                f'{p[j]:.2f}%',
                ha='center',
                va='bottom',
                fontsize=8
            )

    # Remove eixos vazios
    for k in range(i + 1, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    plt.show()



def outliers_iqr(df, cols: list):
    """
    Remove outliers de um DF usando o método IQR (Interquartile Range).
    """
    df_filtrado = df.copy()

    for c in cols:
        if pd.api.types.is_numeric_dtype(df_filtrado[c]):
            Q1 = df_filtrado[c].quantile(0.25)
            Q3 = df_filtrado[c].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df_filtrado = df_filtrado[(df_filtrado[c] >= lower_bound) & (df_filtrado[c] <= upper_bound)]
        else:
            print(f"A coluna '{c}' não é numérica e foi ignorada.")

    return df_filtrado


def plot_categoricas(df, cols=None):
    # Detecta colunas categóricas automaticamente
    if cols is None:
        cols = df.select_dtypes(include=['object', 'category', 'bool', 'int']).columns.tolist()
    
    if len(cols) == 0:
        print("Nenhuma variável categórica identificada.")
        return

    # Paleta padrão
    cores = ['tomato', 'teal', 'orange']

    n_cols = len(cols)
    linhas = math.ceil(n_cols / 3)
    fig, axes = plt.subplots(linhas, 3, figsize=(15, 5 * linhas))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        # Calcula distribuição percentual
        p = df[col].value_counts(normalize=True).sort_index() * 100
        
        # Cores alternadas
        cor = cores[i % len(cores)]

        bars = axes[i].bar(
            p.index.astype(str), p.values,
            color=cor,
            edgecolor='black',
            width=0.6
        )

        axes[i].set_title(f'{col}', fontsize=12)
        axes[i].set_ylabel('Percentual (%)')
        axes[i].set_ylim(0, max(100, p.max() + 10))

        # Rótulos de percentual nas barras
        for bar, percentage in zip(bars, p.values):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{percentage:.1f}%',
                ha='center', va='bottom',
                fontsize=9
            )

    # Remove subplots vazios (caso não seja múltiplo de 3)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
