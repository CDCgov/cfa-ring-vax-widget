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
    duration = 5.0

    times = np.array(
        ringvax.Simulation.generate_infection_delays(
            rng=rng, rate=0.5, infectious_duration=duration
        )
    )

    assert max(times) <= duration
    assert (times.round(3) == np.array([1.209, 1.318, 1.593, 2.205, 4.82])).all()


def test_get_infection_history(rng):
    params = {
        "n_generations": 4,
        "latent_duration": 1.0,
        "infectious_duration": 1.0,
        "infection_rate": 2.0,
    }
    s = ringvax.Simulation(params=params, seed=rng)
    history = s.generate_infection_history(t_exposed=0.0)
    # for ease of testing, make this a list of rounded numbers
    history["t_infections"] = [round(float(x), 3) for x in history["t_infections"]]
    assert history == {
        "t_exposed": 0.0,
        "t_infections": [1.262, 1.319],
        "t_infectious": 1.0,
        "t_recovered": 2.0,
    }


def test_get_infection_history_nonzero(rng):
    """Infection history should shift if exposure time changes"""
    params = {
        "n_generations": 4,
        "latent_duration": 1.0,
        "infectious_duration": 1.0,
        "infection_rate": 2.0,
    }
    s = ringvax.Simulation(params=params, seed=rng)
    history = s.generate_infection_history(t_exposed=10.0)
    assert history == {
        "t_exposed": 10.0,
        "t_infections": [
            np.float64(11.261692423863543),
            np.float64(11.319097058414197),
        ],
        "t_infectious": 11.0,
        "t_recovered": 12.0,
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
    }
    s = ringvax.Simulation(params=params, seed=rng)
    s.run()
    assert len(s.infections) == 278


def test_intervene1_infector_not_infected(rng):
    """If infector is not actually infected, infectee is not actually infected"""
    s = ringvax.Simulation(params={}, seed=rng)
    infector = s.create_person()
    infectee = s.create_person()
    s.update_person(infector, {"actually_infected": False})
    s.update_person(
        infectee,
        {
            "infector": infector,
            "generation": 1,
            "passive_detected": None,
            "t_passive_detected": None,
        },
    )
    s._intervene1(infectee)
    assert s.get_person_property(infectee, "actually_infected") is False
