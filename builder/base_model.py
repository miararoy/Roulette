from time import time

from ehmodelbuilder.save_load_model import ModelFileHandler

class BaseModel(object):
    def __init__(
        self,
    ):  
        self.model_name = str(int(time()))
        self.training_set_y = None
        self.training_set_x = None
        self.validation_set_y = None
        self.validation_set_x = None
        self.validation_prediction_y = None

        self.model = None
        self.error = 1e6
        self._score = 1e6


    def fit(
        self,
        X,
        y,
    ):
        pass


    def predict(
        self,
        X,
    ):
        pass


    def score(
        self,
        y_real,
    ):  
        pass

    def get_params(
        self,
    ):
        try:
            params = self.model.best_params_
        except:
            params = None
        return params

    def get_coef(
        self,
    ):
        try:
            coef = self.model.best_estimator_.coef_
            sigma = self.model.best_estimator_.sigma_
        except:
            coef = None
            sigma = None
        return coef, sigma
    
    def save(
        self,
        base_path,
    ):
        return ModelFileHandler(model=self).save(base_path)

    
    def load(
        self,
        model_path,
    ):
        self.model, self.model_name = ModelFileHandler().load(model_path)

