"""Build the scikit-learn pipeline: impute + one-hot encode, then a regressor."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mp088.cleaning import FEATURES, FEATURES_CAT, FEATURES_NUM


def make_regressor(setting, random_state):
    s = dict(setting)
    name = s.pop("name")
    if name == "ridge":
        return Ridge(**s)
    if name == "random_forest":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **s)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=random_state, **s)
    raise ValueError(f"unknown model: {name}")


def build_pipeline(setting, random_state=42):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])
    return Pipeline([("pre", pre), ("reg", make_regressor(setting, random_state))])


def mae(model, df, target="metres_per_shift"):
    return float(mean_absolute_error(df[target], model.predict(df[FEATURES])))
