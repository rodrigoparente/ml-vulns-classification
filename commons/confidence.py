# third-party imports
import numpy as np
import scipy.stats as st


def interval(data):
    if len(data) <= 30:
        lower, upper = st.t.interval(
            alpha=.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
        return lower, upper

    lower, upper = st.norm.interval(
        alpha=.95, loc=np.mean(data), scale=st.sem(data))

    return lower, upper
