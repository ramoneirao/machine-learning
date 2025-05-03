from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


def modelos_lineares(x_train, y_train, x_test, y_test):
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
    
    return results
