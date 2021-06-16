# Module: preprocess/time_series
# License: MIT
# Purpose: Time series preprocessing


from abc import ABC, abstractmethod
from typing import Union

import pandas as pd


class TimeSeriesPreprocessor(ABC):
    def __init__(self, base_module):
        self.base_module = base_module

    @abstractmethod
    def preprocess(self, data: pd.Series, **kwargs):
        pass

    @abstractmethod
    def impute(self, data: pd.Series, **kwargs):
        pass


class SktimeTSPreprocessor(TimeSeriesPreprocessor):
    from sktime.forecasting.base import BaseForecaster
    from sktime.forecasting.trend import PolynomialTrendForecaster

    def __init__(self):
        super().__init__('sktime')

    def detrend(
            self,
            data: pd.Series,
            forecaster: Union[PolynomialTrendForecaster, BaseForecaster] = None,
            **kwargs
    ) -> pd.Series:
        from sktime.transformations.series.detrend import Detrender
        from sktime.forecasting.trend import PolynomialTrendForecaster

        if forecaster is None:
            forecaster = PolynomialTrendForecaster(
                degree=kwargs.get('degree', 1),
                with_intercept=kwargs.get('with_intercept', True)
            )

        transformer = Detrender(forecaster=forecaster)
        return transformer.fit_transform(data)

    def deseasonalize(self, data: pd.Series, **kwargs):
        pass

    def box_cox_transform(self, *args, **kwargs):
        pass

    def preprocess(self, data: pd.Series, **kwargs):
        pass

    def impute(self, data: pd.Series, **kwargs):
        pass


if __name__ == '__main__':
    from sktime.datasets import load_airline

    ts = load_airline()
    ts_preprocessor = SktimeTSPreprocessor()
    print(ts_preprocessor.detrend(ts))
