from os import path, makedirs
from sklearn.externals import joblib
import pickle
import importlib

class ModelFileHandler(object):
    def __init__(
        self, 
        model=None,
    ):
        self.model = model
        self.path = None


    def save(
        self,
        _dir=None,
    ):  
        model_name = self.model.model_name
        model_directory = path.join(_dir, model_name)
        if not path.exists(model_directory):
            makedirs(model_directory)
        self.path = path.join(model_directory, "{}.joblib".format(model_name))
        try:
            out = self.model.model
            print("trying to pickle {x} of type {xx}".format(
                x=out,
                xx=type(out)
            ))
            with open(self.path, 'wb') as model_file:
                joblib.dump(
                    out,
                    model_file,
                )
            try:
                if self.model.pca:
                    self.path_pca = path.join(model_directory, "{}_pca.joblib".format(model_name))
                    out_pca = self.model.pca
                    with open(self.path_pca, 'wb') as pca_file:
                        joblib.dump(
                            out_pca,
                            pca_file,
                        )
            except:
                print("will not save model PCA")
            return model_directory
        except pickle.PicklingError as e:
            print("Cannot picke model at {p} due to {e}".format(p=self.path, e=e)) 
            return model_directory


    def load(
        self,
        model_path: str,
    ):
        with open(model_path, 'rb') as model_file:
            return joblib.load(model_file), ".".join(model_path.split('/')[-1].split('.')[:-1])


def load_model(
    model_source: str,
):
    # LOAD MODEL
    spec = importlib.util.spec_from_file_location(
        name='model',
        location=path.join(model_source, "model.py")
    )
    model = importlib.util.module_from_spec(
        spec
    )
    spec.loader.exec_module(model)
    return model.Model
