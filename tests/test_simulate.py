import numpy as np
import numpy.random
import pytest

import ringvax


@pytest.fixture
def rng():
    return numpy.random.default_rng(1234)


def test_infection_delays_zero_rate(rng):
    """If zero rate, zero infections"""
    assert (
        list(
            ringvax.Simulation.generate_infection_times(
                rng, rate=0.0, infectious_duration=100.0
            )
        )
        == []
    )


def test_infection_delays_zero_duration(rng):
    """If zero duration, zero infections"""
    assert (
        list(
            ringvax.Simulation.generate_infection_times(
                rng, rate=100.0, infectious_duration=0.0
            )
        )
        == []
    )


def test_infection_delays(rng):
    duration = 5.0

    times = np.array(
        ringvax.Simulation.generate_infection_times(
            rng=rng, rate=0.5, infectious_duration=duration
        )
    )

    assert max(times) <= duration
    assert (times.round(3) == np.array([0.59, 1.209, 1.593, 4.82])).all()


def test_generate_disease_history(rng):
    params = {
        "n_generations": 4,
        "latent_duration": 1.0,
        "infectious_duration": 1.0,
        "infection_rate": 2.0,
    }
    s = ringvax.Simulation(params=params, seed=rng)
    history = s.generate_disease_history(t_exposed=0.0)
    # for ease of testing, make this a list of rounded numbers

    assert history == {
        "t_exposed": 0.0,
        "t_infectious": 1.0,
        "t_recovered": 2.0,
        "infection_rate": 2.0,
    }


def test_generate_disease_history_nonzero(rng):
    """Infection history should shift if exposure time changes"""
    params = {
        "n_generations": 4,
        "latent_duration": 1.0,
        "infectious_duration": 1.0,
        "infection_rate": 2.0,
    }
    s = ringvax.Simulation(params=params, seed=rng)
    history = s.generate_disease_history(t_exposed=10.0)
    assert history == {
        "t_exposed": 10.0,
        "t_infectious": 11.0,
        "t_recovered": 12.0,
        "infection_rate": 2.0,
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
        "active_detection_delay": 2.0,
        "max_infections": 100,
    }
    s = ringvax.Simulation(params=params, seed=rng)
    s.run()
    assert len(s.infections) == 21


def test_simulate_max_infections(rng):
    params = {
        "n_generations": 4,
        "latent_duration": 1.0,
        "infectious_duration": 3.0,
        "infection_rate": 1.0,
        "p_passive_detect": 0.5,
        "passive_detection_delay": 2.0,
        "p_active_detect": 0.15,
        "active_detection_delay": 2.0,
        "max_infections": 10,
    }
    s = ringvax.Simulation(params=params, seed=rng)
    s.run()
    assert len(s.infections) == 10
