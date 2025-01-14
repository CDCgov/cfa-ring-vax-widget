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


def test_generational_counts():
    params = {
        "n_generations": 6,
        "latent_duration": 1.0,
        "infectious_duration": 3.0,
        "infection_rate": 0.5,
        "p_passive_detect": 0.5,
        "passive_detection_delay": 2.0,
        "p_active_detect": 0.15,
        "active_detection_delay": 2.0,
        "max_infections": 1000000,
    }

    n_sims = 3
    sims = []
    for i in range(n_sims):
        sims.append(ringvax.Simulation(params=params, rng=np.random.default_rng(i)))
        sims[-1].run()

    all_sims = ringvax.summary.get_all_person_properties(sims)
    max_obs_gen = [
        max(sim.get_person_property(id, "generation") for id in sim.infections)
        for sim in sims
    ]
    obs_g_max = max(max_obs_gen)

    gen_counts = ringvax.summary.get_infection_counts_by_generation(all_sims)

    assert gen_counts.shape[0] == (obs_g_max + 1) * n_sims

    for i, sim in enumerate(sims):
        sim_counts = gen_counts.filter(pl.col("simulation") == i)
        assert sim_counts.shape[0] == obs_g_max + 1

        for g in range(max_obs_gen[i] + 1):
            assert sim_counts.filter(pl.col("generation") == g)["num_infections"][
                0
            ] == len(sim.query_people({"generation": g}))

        for g in range(max_obs_gen[i] + 1, obs_g_max):
            assert (
                sim_counts.filter(pl.col("generation") == g)["num_infections"][0] == 0
            )
