from typing import Union

import numpy as np


class MeanWindowForecastingModel:

    def __init__(
        self,
        name: str,
        target_name: str,
        **kwargs
    ):
        self.name = name
        self.target_name = target_name

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        # maintain the same interface as the other models
        pass

    def predict(self, X: 'pd.DataFrame'):
        """Forecast the cn2 using the mean of the lagged values."""
        # predict the mean for each entry in X
        # X contains some number of lagged values of the target variable
        # we will use the mean of these lagged values as our prediction
        X = X[[c for c in X.columns if c.startswith(self.target_name)]]

        # develop a prediction for each row in X
        preds = []
        for i in range(len(X)):
            # fit a line to the lagged values
            pred = np.mean(X.iloc[i, :].values)
            preds.append(pred)

        return np.array(preds)
