import pandas as pd
from typing import List, Tuple


class PDF(object):

    def __init__(self, experiment: pd.DataFrame):
        self.experiment = experiment

    def __call__(self, variables: List[str]):
        pass

    def given(self, variables: List[str]) -> 'ConditionalProbability':
        return ConditionalProbability(self.experiment, variables)


class ConditionalProbability(object):

    def __init__(self, experiment, variables):
        group_by = experiment.groupby(variables).sum()

        self.conditional_experiment = group_by / group_by.sum()

    def __getitem__(self, item: Tuple):
        """
        :param item: Tuple of values ordered by the original variables?
        """
        return self.conditional_experiment.loc[item]
