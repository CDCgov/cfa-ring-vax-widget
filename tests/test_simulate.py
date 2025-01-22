import json
from pathlib import Path

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
        ringvax.Simulation.generate_infection_waiting_times(
            rng, rate=0.0, infectious_duration=100.0
        ).size
        == 0
    )


def test_infection_delays_zero_duration(rng):
    """If zero duration, zero infections"""
    assert (
        ringvax.Simulation.generate_infection_waiting_times(
            rng, rate=100.0, infectious_duration=0.0
        ).size
        == 0
    )


def test_infection_delays(rng):
    duration = 5.0

    times = np.array(
        ringvax.Simulation.generate_infection_waiting_times(
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
    s = ringvax.Simulation(params=params, rng=rng)
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
    s = ringvax.Simulation(params=params, rng=rng)
    history = s.generate_disease_history(t_exposed=10.0)
    assert history == {
        "t_exposed": 10.0,
        "t_infectious": 11.0,
        "t_recovered": 12.0,
        "infection_rate": 2.0,
    }


@pytest.fixture
def base_params():
    return {
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


def test_simulate(rng, base_params):
    s = ringvax.Simulation(params=base_params, rng=rng)
    s.run()
    assert len(s.infections) == 29


def test_simulate_max_infections(rng, base_params):
    params = base_params
    params["max_infections"] = 10
    s = ringvax.Simulation(params=params, rng=rng)
    s.run()
    assert len(s.infections) == 10


def test_simulate_set_field(rng, base_params):
    s = ringvax.Simulation(params=base_params, rng=rng)
    id = s.create_person()
    s.update_person(id, {"generation": 0})
    assert s.get_person_property(id, "generation") == 0


def test_simulate_error_on_bad_get_property(rng, base_params):
    s = ringvax.Simulation(params=base_params, rng=rng)
    id = s.create_person()

    with pytest.raises(RuntimeError, match="foo"):
        s.get_person_property(id, "foo")


def test_simulate_error_on_bad_update_property(rng, base_params):
    s = ringvax.Simulation(params=base_params, rng=rng)
    id = s.create_person()

    with pytest.raises(RuntimeError, match="foo"):
        s.update_person(id, {"foo": 0})


def test_snapshot(rng):
    params = {
        "n_generations": 4,
        "latent_duration": 1.0,
        "infectious_duration": 5.0,
        "infection_rate": 1.0,
        "p_passive_detect": 0.5,
        "passive_detection_delay": 4.0,
        "p_active_detect": 0.5,
        "active_detection_delay": 2.0,
        "max_infections": 100,
    }
    s = ringvax.Simulation(params=params, rng=rng)
    s.run()

    for x in s.infections.values():
        x["infection_times"] = x["infection_times"].tolist()

    with open(Path("tests", "data", "snapshot.json")) as f:
        snapshot = json.load(f)

    assert s.infections == snapshot


class TestResolveDetectionHistory:
    @pytest.fixture
    @staticmethod
    def kwargs():
        """Baseline keyword arguments for resolve_detection_history tests"""
        return {
            "potentially_passive_detected": False,
            "potentially_active_detected": False,
            "passive_detection_delay": 5.0,
            "active_detection_delay": 2.0,
            "t_exposed": 0.0,
            "t_recovered": 10.0,
            "t_infector_detected": None,
        }

    def test_baseline(self, kwargs):
        """No potential detections"""
        assert ringvax.Simulation.resolve_detection_history(**kwargs) == {
            "detected": False,
            "t_detected": None,
            "detect_method": None,
        }

    def test_passive_only(self, kwargs):
        """Passive detection only"""
        kwargs["potentially_passive_detected"] = True
        assert ringvax.Simulation.resolve_detection_history(**kwargs) == {
            "detected": True,
            "t_detected": 5.0,
            "detect_method": "passive",
        }

    def test_passive_bad_time(self, kwargs):
        """Passive detection after recovery"""
        kwargs["potentially_passive_detected"] = True
        kwargs["passive_detection_delay"] = 11.0
        assert ringvax.Simulation.resolve_detection_history(**kwargs) == {
            "detected": False,
            "t_detected": None,
            "detect_method": None,
        }

    def test_active_only(self, kwargs):
        """Active detection only"""
        kwargs["potentially_active_detected"] = True
        kwargs["t_infector_detected"] = 0.0
        assert ringvax.Simulation.resolve_detection_history(**kwargs) == {
            "detected": True,
            "t_detected": 0.0 + 2.0,
            "detect_method": "active",
        }

    def test_active_bad_time(self, kwargs):
        """Active detection after recovery"""
        kwargs["potentially_active_detected"] = True
        kwargs["t_infector_detected"] = 5.0
        kwargs["active_detection_delay"] = 6.0
        assert ringvax.Simulation.resolve_detection_history(**kwargs) == {
            "detected": False,
            "t_detected": None,
            "detect_method": None,
        }

    def test_both_passive_wins(self, kwargs):
        """Both passive and active detection, passive wins"""
        kwargs["potentially_passive_detected"] = True
        kwargs["potentially_active_detected"] = True
        kwargs["t_infector_detected"] = 5.0
        assert ringvax.Simulation.resolve_detection_history(**kwargs) == {
            "detected": True,
            "t_detected": 5.0,
            "detect_method": "passive",
        }

    def test_both_active_wins(self, kwargs):
        """Both passive and active detection, active wins"""
        kwargs["potentially_passive_detected"] = True
        kwargs["potentially_active_detected"] = True
        kwargs["t_infector_detected"] = 1.0
        assert ringvax.Simulation.resolve_detection_history(**kwargs) == {
            "detected": True,
            "t_detected": 1.0 + 2.0,
            "detect_method": "active",
        }

    def test_both_neither(self, kwargs):
        """Both passive and active detection, neither wins"""
        kwargs["potentially_passive_detected"] = True
        kwargs["potentially_active_detected"] = True
        kwargs["t_infector_detected"] = 9.0
        kwargs["passive_detection_delay"] = 11.0
        assert ringvax.Simulation.resolve_detection_history(**kwargs) == {
            "detected": False,
            "t_detected": None,
            "detect_method": None,
        }
