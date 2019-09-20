from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from bliz.builder.base_model import BaseModel


tuned_params = {
    "n_estimators": [10, 20],
    "max_depth": [3,5],
    "min_samples_split": [2,5]
}

class Model(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.model = GridSearchCV(
            RandomForestRegressor(),
            tuned_params,
            cv=5
        )

    def fit(self,X,y):
        self.model.fit(X,y)
    
    def predict(self, X):
        return self.model.predict(X)
    
