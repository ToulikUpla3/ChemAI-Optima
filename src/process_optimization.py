from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def optimize_chemical_process(data):
    """Train a model to optimize chemical processes."""
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Target (e.g., yield)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Model R^2 Score: {score}")
    return model
