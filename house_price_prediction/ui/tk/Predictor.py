import json
import zipfile

import pickle


def save_model(model, feature_names, zip_path):
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('model.pkl', pickle.dumps(model))
        zf.writestr('feature_names.json', json.dumps(list(feature_names)))


def load_model(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        model = pickle.loads(zf.read('model.pkl'))
        fns = json.loads(zf.read('feature_names.json').decode('utf-8'))
        return model, fns


def load_model_any(path, default_feature_names=None):
    """Load model from either our zip format or a plain pickle file.

    - If `path` is a zip file, tries to read `model.pkl` and optional
      `feature_names.json`. If feature names are missing, falls back to
      `default_feature_names`.
    - If `path` is not a zip, attempts to `pickle.load` the file directly
      (supports the style shown in docs: pickle.dump(lm, open('*.zip','wb'))).
    """
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, 'r') as zf:
            names = zf.namelist()
            # model
            if 'model.pkl' in names:
                model = pickle.loads(zf.read('model.pkl'))
            else:
                # try first pickle-like entry
                pickles = [n for n in names if n.endswith(('.pkl', '.pickle', '.bin', '.model'))]
                if not pickles:
                    raise KeyError('model.pkl not found in zip')
                model = pickle.loads(zf.read(pickles[0]))
            # feature names
            if 'feature_names.json' in names:
                fns = json.loads(zf.read('feature_names.json').decode('utf-8'))
            else:
                fns = list(default_feature_names or [])
            return model, fns
    # plain pickle file
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model, list(default_feature_names or [])