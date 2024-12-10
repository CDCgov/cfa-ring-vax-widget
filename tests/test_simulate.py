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
            ringvax.Simulation.get_infection_delays(
                rng, rate=0.0, infectious_duration=1.0
            )
        )
        == []
    )


def test_infection_delays_zero_duration(rng):
    assert (
        list(
            ringvax.Simulation.get_infection_delays(
                rng, rate=1.0, infectious_duration=0.0
            )
        )
        == []
    )


def test_infection_delays(rng):
    duration = 10.0

    times = np.array(
        ringvax.Simulation.get_infection_delays(
            rng=rng, rate=2.0, infectious_duration=duration
        )
    )

    assert max(times) <= duration

    assert (
        times.round(3)
        == np.array(
            [1.524, 2.238, 4.051, 4.864, 5.847, 6.268, 6.456, 6.823, 8.374, 9.315]
        )
    ).all()


def test_infection_delays_short(rng):
    """If using the same rng, shorter period should mean truncating only"""
    duration = 5.0
    times = np.array(
        ringvax.Simulation.get_infection_delays(
            rng=rng, rate=2.0, infectious_duration=duration
        )
    )

    assert max(times) <= duration

    assert (times.round(3) == np.array([1.524, 2.238, 4.051, 4.864])).all()


def test_get_infection_history():
    params = {
        "n_generations": 4,
        "latent_duration": 1.0,
        "infectious_duration": 1.0,
        "infection_rate": 1.0,
    }
    s = ringvax.Simulation(params=params, seed=1234)
    history = s.get_infection_history(t_exposed=0.0)
    assert history == {
        "t_exposed": 0.0,
        "t_infections": [np.float64(1.714650803637824)],
        "t_infectious": 1.0,
        "t_recovered": 2.0,
    }


def test_simulate():
    params = {
        "n_generations": 4,
        "latent_duration": 1.0,
        "infectious_duration": 3.0,
        "infection_rate": 1.0,
    }
    s = ringvax.Simulation(params=params, seed=1234)
    s.run()
    assert len(s.infections) == 73
