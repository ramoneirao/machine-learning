from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


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


def verifica_faltantes(df):
    """
    Verifica se há valores faltantes no DataFrame.
    Retorna o valor em porcentagem de cada coluna.
    """
    return df.isnull().sum() / len(df) * 100