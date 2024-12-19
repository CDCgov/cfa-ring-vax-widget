import types

import numpy as np
import polars as pl

import ringvax
import ringvax.summary


def test_prep_for_df():
    infection = {"infection_times": np.array([0, 1, 2]), "detected": False}
    assert ringvax.summary._prepare_for_df(infection) == {
        "infection_times": [0, 1, 2],
        "detected": False,
    }


def test_get_all_person_properties():
    infections = {
        "index": {
            "infector": None,
            "generation": 0,
            "t_exposed": 0.0,
            "t_infectious": 1.0,
            "t_recovered": 2.0,
            "infection_rate": 0.5,
            "detected": False,
            "detect_method": None,
            "t_detected": None,
            "infection_times": np.array([1.1, 1.2, 1.3]),
        },
        "first_person": {
            "infector": "index",
            "generation": 1,
            "t_exposed": 1.1,
            "t_infectious": 2.1,
            "t_recovered": 3.1,
            "infection_rate": 0.5,
            "detected": True,
            "detect_method": "passive",
            "t_detected": 1.5,
            "infection_times": np.array(()),
        },
    }
    params = {"n_generations": 1, "max_infections": 10}
    termination = {"criterion": "extinction"}
    sim1 = types.SimpleNamespace(
        params=params, infections=infections, termination=termination
    )
    sim2 = types.SimpleNamespace(
        params=params, infections=infections, termination=termination
    )

    x = ringvax.summary.get_all_person_properties([sim1, sim2])  # type: ignore

    # result should be a data frame of length 4
    assert isinstance(x, pl.DataFrame)
    assert x.shape[0] == 4
