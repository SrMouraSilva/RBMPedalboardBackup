import pandas as pd
from discrete_probability.PDF import PDF
from typing import List


class Counselor(object):
    """
    Counselor suggests plugins based in your knowledge
    """

    def __init__(self, bag_of_plugins: pd.DataFrame):
        self.P = PDF(bag_of_plugins)

    def suggest(self, plugins: List[str]) -> pd.Series:
        """
        :return: Plugins that combine with the parameter plugins
        """
        all_true = (1, ) * len(plugins)

        return self.P.given(plugins)[all_true].sort_values(ascending=False)
