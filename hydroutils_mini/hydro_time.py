import datetime
from typing import Union
import numpy as np


def t2str(t_: Union[str, datetime.datetime]):
    if type(t_) is str:
        return datetime.datetime.strptime(t_, "%Y-%m-%d")
    elif type(t_) is datetime.datetime:
        return t_.strftime("%Y-%m-%d")
    else:
        raise NotImplementedError("We don't support this data type yet")


def t_range_days(t_range, *, step=np.timedelta64(1, "D")) -> np.array:
    """
    Transform the two-value t_range list to a uniformly-spaced list (default is a daily list).
    For example, ["2000-01-01", "2000-01-05"] -> ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]
    Parameters
    ----------
    t_range
        two-value t_range list
    step
        the time interval; its default value is 1 day
    Returns
    -------
    np.array
        a uniformly-spaced (daily) list
    """
    startdata = datetime.datetime.strptime(t_range[0], "%Y-%m-%d")
    enddata = datetime.datetime.strptime(t_range[1], "%Y-%m-%d")
    return np.arange(startdata, enddata, step)
