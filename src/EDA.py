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


def categoricasXtarget(df, target, cols=None, titulo="Relações com variável alvo"):
    # Detectar colunas categóricas automaticamente se não forem passadas
    if cols is None:
        cols = df.select_dtypes(include=['object', 'category', 'int', 'bool']).columns.tolist()
        cols = [c for c in cols if c != target]
    
    if not cols:
        print("Nenhuma variável categórica encontrada para plotar.")
        return

    # Definir paleta de cores padrão
    cores = ['tomato', 'teal', 'orange']

    # Layout automático
    n = len(cols)
    linhas = math.ceil(n / 3)
    fig, axes = plt.subplots(linhas, 3, figsize=(6 * 3, 5 * linhas))
    axes = np.array(axes).flatten()

    # Loop sobre as variáveis
    for i, col in enumerate(cols):
        # Agrupamento
        df_grouped = (
            df.groupby([col, target])
              .size()
              .reset_index(name='count')
        )
        df_grouped['percent'] = df_grouped['count'] * 100 / len(df)

        # Valores únicos da variável
        categorias = sorted(df[col].dropna().unique())
        x = np.arange(len(df[target].unique()))
        wd = 0.8 / len(categorias)  # largura adaptativa

        # Plot
        for j, cat in enumerate(categorias):
            y = df_grouped.query(f"{col} == @cat")['percent'].tolist()
            axes[i].bar(x + (j - len(categorias)/2)*wd, y, wd,
                        color=cores[j % len(cores)],
                        edgecolor='black', label=str(cat))
            
            # Adiciona rótulos de percentual
            for k in range(len(x)):
                axes[i].text(
                    x[k] + (j - len(categorias)/2)*wd,
                    y[k],
                    f'{y[k]:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

        # Ajustes de eixo e título
        axes[i].set_title(f"{col} x {target}")
        axes[i].set_xlabel(target)
        axes[i].set_ylabel("Percentual (%)")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(df[target].unique())
        axes[i].set_ylim(0, max(60, df_grouped['percent'].max() + 10))
        axes[i].legend(title=col, fontsize=8)

    # Remover subplots vazios
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def correlacao(df, metodo='pearson'):
    """
    Plota a matriz de correlação com mapa de calor.
    """
    plt.figure(figsize=(12, 8))
    corr = df.corr(method=metodo)
    corr[(corr < 0.15)&(corr > -0.15)] = pd.NA
    corr[corr >=0.99] = pd.NA
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title(f'Matriz de Correlação ({metodo.capitalize()})', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

