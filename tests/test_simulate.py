import copy

import numpy as np
import numpy.random
import pytest

import ringvax


@pytest.fixture
def rng():
    return numpy.random.default_rng(1234)


def test_infection_delays_zero_rate(rng):
    assert (
        list(
            ringvax.Simulation.generate_infection_delays(
                rng, rate=0.0, infectious_duration=1.0
            )
        )
        == []
    )


def test_infection_delays_zero_duration(rng):
    assert (
        list(
            ringvax.Simulation.generate_infection_delays(
                rng, rate=1.0, infectious_duration=0.0
            )
        )
        == []
    )


def test_infection_delays(rng):
    rng2 = copy.deepcopy(rng)

    times = np.array(
        ringvax.Simulation.generate_infection_delays(
            rng=rng, rate=0.5, infectious_duration=10.0
        )
    )

    assert max(times) <= 10.0
    assert (times.round(3) == np.array([3.047, 4.477, 8.103, 9.728])).all()

    # If using the same rng, shorter period should mean truncating only
    times2 = np.array(
        ringvax.Simulation.generate_infection_delays(
            rng=rng2, rate=0.5, infectious_duration=5.0
        )
    )

    assert max(times2) <= 5.0
    assert (times2 == times[times < 5.0]).all()


def test_get_infection_history(rng):
    params = {
        "n_generations": 4,
        "latent_duration": 1.0,
        "infectious_duration": 1.0,
        "infection_rate": 2.0,
    }
    s = ringvax.Simulation(params=params, seed=rng)
    history = s.generate_infection_history(t_exposed=0.0)
    assert history == {
        "t_exposed": 0.0,
        "t_infections": [np.float64(1.7618651221460064)],
        "t_infectious": 1.0,
        "t_recovered": 2.0,
    }


def test_simulate(rng):
    params = {
        "n_generations": 4,
        "latent_duration": 1.0,
        "infectious_duration": 3.0,
        "infection_rate": 1.0,
        "p_passive_detect": 0.5,
        "passive_detection_delay": 2.0,
        "p_active_detect": 0.15,
    }
    s = ringvax.Simulation(params=params, seed=rng)
    s.run()
    assert len(s.infections) == 72
