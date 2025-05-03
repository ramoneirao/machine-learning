from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, cross_val_score
import numpy as np


def modelos_lineares(x_train, y_train, x_test, y_test, melhor_modelo=False):
    """
    Treina e avalia múltiplos modelos de regressão linear.
    Retorna um dicionário com os R² (coeficiente de determinação) de cada modelo.
    """
    # Inicializar os modelos com hiperparâmetros
    modelos = {
        'Regressão Linear Simples': LinearRegression(),
        'Regressão Ridge': Ridge(),
        'Regressão Lasso': Lasso(),
        'Regressão ElasticNet': ElasticNet()
    }

    # Avalia os modelos
    results = {}
    for name, model in modelos.items():
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        results[name] = score
    
    # Se melhor_modelo for True, retorna o modelo com o maior R²
    if melhor_modelo:
            melhor = max(results, key=results.get)
            return print(f"Melhor modelo: {melhor} com R² = {results[melhor]}")

    # Retorna todos os resultados
    print("Resultados dos modelos:")
    for name, score in results.items():
        print(f"{name}: R² = {score}")


def modelos_lineares_kf(x, y, melhor_modelo=False):
    """
    Treina e avalia múltiplos modelos de regressão linear utilizando KFold.
    Retorna um dicionário com as médias dos R² (coeficiente de determinação) de cada modelo.
    """
    # Inicializar os modelos com hiperparâmetros
    modelos = {
        'Regressão Linear Simples': LinearRegression(),
        'Regressão Ridge': Ridge(),
        'Regressão Lasso': Lasso(),
        'Regressão ElasticNet': ElasticNet()
    }

    # Configuração do KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Avalia os modelos
    results = {}
    for name, model in modelos.items():
        scores = cross_val_score(model, x, y, cv=kf)
        results[name] = np.mean(scores)

    # Se melhor_modelo for True, retorna o modelo com o maior R² médio
    if melhor_modelo:
        melhor = max(results, key=results.get)
        return print(f"Melhor modelo: {melhor} com R² médio = {results[melhor]:.4f}")

    # Retorna todos os resultados
    print("Resultados dos modelos:")
    for name, score in results.items():
        print(f"{name}: R² médio = {score:.4f}")


def verifica_faltantes(df):
    """
    Verifica se há valores faltantes no DataFrame.
    Retorna o valor em porcentagem de cada coluna.
    """
    return df.isnull().sum() / len(df) * 100