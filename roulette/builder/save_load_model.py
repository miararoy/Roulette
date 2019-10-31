import os
import sys
import joblib
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
        path=None,
    ):
        model_name = self.model.model_name
        model_dir = os.path.join(path, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, "{}.joblib".format(model_name))
        try:
            out = self.model
            print("trying to pickle {} of type {}".format(out, type(out)))
            with open(model_path, 'wb') as model_file:
                joblib.dump(
                    out,
                    model_file,
                )
            return model_dir
        except pickle.PicklingError as e:
            print("Cannot picke model at {p} due to {e}".format(
                p=model_path, e=e))
            return model_dir

    def load(
        self,
        model_dir: str,
    ):
        model_name = "{}.joblib".format(os.path.split(model_dir)[-1])
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            raise FileExistsError("file {} is not found in folder {} ".format(
                model_name, model_dir
            ) + "cannot load")
        with open(model_path, 'rb') as model_file:
            return joblib.load(model_file)


def load_model(
    model_source: str,
):
    # LOAD MODEL
    spec = importlib.util.spec_from_file_location(
        name='model',
        location=os.path.join(model_source, "model.py")
    )
    model = importlib.util.module_from_spec(
        spec
    )
    spec.loader.exec_module(model)
    sys.modules['model'] = model
    return model.Model
