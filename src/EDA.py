"""
Funções para Análise Exploratória de Dados (EDA)
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_boxplot(df, target=None, columns=None):
    """
    Plota um boxplot para as colunas especificadas do DataFrame em subplots.
    
    Parâmetros:
    df: DataFrame com os dados
    target: coluna target para segmentar os boxplots (opcional)
    columns: lista de colunas específicas para plotar (opcional)
             Se None, usa todas as colunas numéricas
    """
    # Se colunas específicas foram fornecidas, usar apenas elas
    if columns is not None:
        # Verificar se as colunas existem no DataFrame
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            print(f"Colunas não encontradas no DataFrame: {missing_cols}")
            return
        
        # Verificar se as colunas são numéricas
        num_cols = []
        for col in columns:
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32', 'int8', 'int16']:
                num_cols.append(col)
            else:
                print(f"Aviso: Coluna '{col}' não é numérica e será ignorada.")
        
        num_cols = list(num_cols)
    else:
        # Selecionar apenas colunas numéricas automaticamente
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Remover a coluna target das colunas numéricas se ela existir
    if target and target in num_cols:
        num_cols.remove(target)
    
    n_cols = len(num_cols)
    
    if n_cols == 0:
        print("Nenhuma coluna numérica válida encontrada!")
        return
    
    # Configurar o layout dos subplots
    max_cols_per_row = 4
    n_rows = (n_cols + max_cols_per_row - 1) // max_cols_per_row  # Arredonda para cima
    n_cols_grid = min(n_cols, max_cols_per_row)
    
    # Criar a figura e os subplots
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(4*n_cols_grid, 4*n_rows))
    
    # Garantir que axes seja sempre um array 2D para facilitar a indexação
    if n_rows == 1 and n_cols_grid == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols_grid == 1:
        axes = [[ax] for ax in axes]
    
    # Plotar cada boxplot
    for i, col in enumerate(num_cols):
        row = i // max_cols_per_row
        col_idx = i % max_cols_per_row
        
        if target:
            # Boxplot segmentado pela variável target
            sns.boxplot(data=df, x=target, y=col, ax=axes[row][col_idx])
            axes[row][col_idx].set_title(f'{col} X {target}')
        else:
            # Boxplot simples
            sns.boxplot(data=df, y=col, ax=axes[row][col_idx])
            axes[row][col_idx].set_title(f'{col}')
        
        # Rotacionar labels do eixo x se necessário
        # axes[row][col_idx].tick_params(axis='x', rotation=45)
    
    # Remover subplots vazios (se houver)
    for i in range(n_cols, n_rows * n_cols_grid):
        row = i // max_cols_per_row
        col_idx = i % max_cols_per_row
        fig.delaxes(axes[row][col_idx])
    
    plt.tight_layout()
    plt.show()


def plot_histogram():
    pass


def outliers_iqr(df,col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df

