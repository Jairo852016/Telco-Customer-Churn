# preprocessors.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ============== PIPELINE NUMÉRICO ==============
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

# ============== CUSTOM ONE HOT ==============
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        try:
            self._oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            self._oh = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self._columns = []

    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object']).copy()
        if X_cat.shape[1] == 0:
            self._columns = []
            self._oh.fit(pd.DataFrame(index=X.index))
            return self
        self._oh.fit(X_cat)
        self._columns = self._oh.get_feature_names_out(X_cat.columns)
        return self

    def transform(self, X, y=None):
        X_cat = X.select_dtypes(include=['object']).copy()
        if X_cat.shape[1] == 0:
            return pd.DataFrame(index=X.index)
        X_cat_oh = self._oh.transform(X_cat)
        return pd.DataFrame(X_cat_oh, columns=self._columns, index=X.index)

# ============== PRE-IMPUTER CATEGÓRICAS ==============
class CustomPreImputer(BaseEstimator, TransformerMixin):
    """
    - Imputa TODAS las columnas categóricas (dtype 'object') con la moda.
    - Para columnas categóricas con EXACTAMENTE 2 valores distintos,
      las convierte a 0 y 1.
    - El resto de categóricas se dejan como texto para OneHotEncoder.
    """

    def __init__(self):
        self.cat_modes_ = {}
        self.binary_mappings_ = {}

    def fit(self, X, y=None):
        X_df = X.copy()
        cat_cols = X_df.select_dtypes(include=["object"]).columns

        for col in cat_cols:
            mode_col = X_df[col].dropna().mode()
            if len(mode_col) > 0:
                self.cat_modes_[col] = mode_col.iloc[0]

            temp = X_df[col].fillna(self.cat_modes_[col]) if col in self.cat_modes_ else X_df[col]
            uniques = list(pd.Series(temp).unique())

            if len(uniques) == 2:
                if set(uniques) == {"Yes", "No"}:
                    mapping = {"No": 0, "Yes": 1}
                else:
                    mapping = {uniques[0]: 0, uniques[1]: 1}
                self.binary_mappings_[col] = mapping

        return self

    def transform(self, X, y=None):
        X_df = X.copy()

        for col, mode_val in self.cat_modes_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(mode_val)

        for col, mapping in self.binary_mappings_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].map(mapping).astype("Int64")

        return X_df

# ============== DATAFRAME PREPARER ==============
class DataFramePreparer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._pre_imputer = CustomPreImputer()
        self._full_pipeline = None
        self._columns = None
        self.input_features_ = None

    def fit(self, X, y=None):
        # 0) Pre-imputación global
        X1 = self._pre_imputer.fit_transform(X)
        self.input_features_ = list(X1.columns)

        # 1) Detección numéricas vs categóricas
        num_attribs = list(X1.select_dtypes(exclude=['object']).columns)
        cat_attribs = list(X1.select_dtypes(include=['object']).columns)

        # 2) ColumnTransformer
        self._full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", CustomOneHotEncoder(), cat_attribs),
        ])
        self._full_pipeline.fit(X1)

        # 3) Nombres de columnas de salida
        out_cols = []
        out_cols.extend(num_attribs)
        cat_encoder = self._full_pipeline.named_transformers_["cat"]
        if hasattr(cat_encoder, "_columns") and len(cat_encoder._columns) > 0:
            out_cols.extend(list(cat_encoder._columns))
        self._columns = out_cols
        return self

    def transform(self, X, y=None):
        X1 = self._pre_imputer.transform(X)
        X_prep = self._full_pipeline.transform(X1)
        return pd.DataFrame(X_prep, columns=self._columns, index=X.index)
