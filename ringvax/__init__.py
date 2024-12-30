import bisect
from typing import Any, List, Optional

import numpy as np
import numpy.random


class Simulation:
    # All infections have this
    INIT_SCHEMA = {
        "id": str | None,
        "infector": str,
        "t_exposed": float,
        "generation": int,
    }

    # These are all None until an infection's history is simulated
    SIM_SCHEMA = {
        "simulated": bool,
        "infectees": list[str],
        "t_infectious": Optional[float],
        "t_infectious_counterfactual": float,
        "t_recovered": Optional[float],
        "t_recovered_counterfactual": float,
        "infection_rate": float,
        "detected": bool,
        "detect_method": Optional[str],
        "t_detected": Optional[float],
    }

    SCHEMA = INIT_SCHEMA | SIM_SCHEMA

    PROPERTIES = set(SCHEMA)

    def __init__(
        self, params: dict[str, Any], rng: Optional[numpy.random.Generator] = None
    ):
        self.params = params
        self.rng = rng if rng is not None else numpy.random.default_rng()
        self.infections = {}
        self.termination: Optional[str] = None

    def instantiate_infection(
        self, infector: str | None, t_exposed: float, generation: int
    ) -> str:
        """Add a new person to the data"""
        id = str(len(self.infections))
        self.infections[id] = {x: None for x in self.SIM_SCHEMA}
        self.infections[id]["id"] = id
        self.infections[id]["infector"] = infector
        self.infections[id]["t_exposed"] = t_exposed
        self.infections[id]["generation"] = generation
        return id

    def update_person(self, id: str, content: dict[str, Any]) -> None:
        bad_properties = set(content.keys()) - self.PROPERTIES
        if len(bad_properties) > 0:
            raise RuntimeError(f"Properties not in schema: {bad_properties}")
        bad_types = set(
            k for k, v in content.items() if not isinstance(v, self.SCHEMA[k])
        )
        if len(bad_types) > 0:
            raise RuntimeError(f"Properties with type not matching schema: {bad_types}")
        self.infections[id] |= content

    def get_person_property(self, id: str, property: str) -> Any:
        """Get a property of a person"""
        if property not in self.PROPERTIES:
            raise RuntimeError(f"Property '{property}' not in schema")

        if id not in self.infections:
            raise RuntimeError(f"No person with {id=}")
        elif property not in self.infections[id]:
            raise RuntimeError(f"Person {id=} does not have property '{property}'")

        return self.infections[id][property]

    def query_people(self, query: Optional[dict[str, Any]] = None) -> List[str]:
        """Get IDs of people with a given set of properties"""
        if query is None:
            return list(self.infections.keys())
        else:
            return [
                id
                for id, person in self.infections.items()
                if all(person[k] == v for k, v in query.items())
            ]

    def run(self) -> None:
        """Run simulation"""
        # queue is pairs (t_exposed, infector)
        # start with the index infection
        infection_queue: List[str] = [self.instantiate_infection(None, 0.0, 0)]

        passed_max_generations = False

        while True:
            # in each pass through this loop, we:
            # - exit the loop if needed
            # - pop one infection off the queue and instantiate it
            # - potentially add that infection's infectees to the queue
            n_infections = len(self.query_people())

            # check if we need to stop the loop
            if len(infection_queue) == 0:
                # no infections left in the queue
                # assign reason for termination
                self.termination = (
                    "max_generations" if passed_max_generations else "extinct"
                )
                # exit the loop
                break
            elif n_infections == self.params["max_infections"]:
                # we are at maximum number of infections
                self.termination = "max_infections"
                # exit the loop
                break
            elif n_infections > self.params["max_infections"]:
                # this loop instantiates infections one at a time. we should
                # exactly hit the maximum and not exceed it.
                raise RuntimeError("Maximum number of infections exceeded")

            # find the person who is infected next, simulate the course of their infection
            # (the queue is time-sorted, so this is the temporally next infection)
            id = infection_queue.pop(0)
            self.simulate_infection(id=id)

            # instantiate infections caused by this infection, collect their IDs
            secondary_ids = self.instantiate_secondary_infections(id=id)

            # if the infector is in the final generation, do not add their
            # infectees to the queue
            generation = self.get_person_property(id, "generation")
            if generation == self.params["n_generations"]:
                passed_max_generations = True
            elif generation > self.params["n_generations"]:
                # this loop instantiates infections one at a time. we should
                # exactly hit the maximum generations and not exceed it.
                raise RuntimeError("Generation count exceeded")
            else:
                # only add infectees to the queue if we are not yet at maximum
                # number of generations
                for id in secondary_ids:
                    bisect.insort_right(
                        infection_queue,
                        id,
                        key=lambda x: self.get_person_property(x, "t_exposed"),
                    )

    def simulate_infection(
        self,
        id: str,
    ) -> None:
        """
        Generate a single infected person's biological disease history, detection
        history and transmission history.

        Returns id
        """
        # keep track of generations
        t_exposed = self.get_person_property(id, "t_exposed")

        # disease state history in this individual
        disease_history = self.simulate_undetected_disease_history(t_exposed=t_exposed)
        self.update_person(id, disease_history)

        # whether this person was detected
        detection_history = self.simulate_detection_history(id)
        self.update_person(id, detection_history)

        # adjust disease history for detection
        t_infectious = disease_history["t_infectious_counterfactual"]
        if (
            detection_history["detected"]
            and detection_history["t_detected"] < t_infectious
        ):
            t_infectious = None

        t_recovered = disease_history["t_recovered_counterfactual"]
        if (
            detection_history["detected"]
            and detection_history["t_detected"] < t_recovered
        ):
            t_recovered = detection_history["t_detected"]

        self.update_person(
            id,
            {
                "simulated": True,
                "t_infectious": t_infectious,
                "t_recovered": t_recovered,
            },
        )

    def simulate_undetected_disease_history(self, t_exposed: float) -> dict[str, Any]:
        """Generate infection history for a single infected person"""
        latent_duration = self.generate_latent_duration()
        infectious_duration = self.generate_infectious_duration()
        infection_rate = self.generate_infection_rate()

        t_infectious = t_exposed + latent_duration
        t_recovered = t_infectious + infectious_duration

        return {
            "t_exposed": t_exposed,
            "t_infectious_counterfactual": t_infectious,
            "t_recovered_counterfactual": t_recovered,
            "infection_rate": infection_rate,
        }

    def simulate_detection_history(self, id: str) -> dict[str, Any]:
        """Determine if a person is infected, and when"""
        infector = self.get_person_property(id, "infector")

        detected = False
        detect_method = None
        t_detected = None

        passive_detected = self.bernoulli(self.params["p_passive_detect"])
        if passive_detected:
            detected = True
            detect_method = "passive"
            t_detected = (
                self.get_person_property(id, "t_exposed")
                + self.generate_passive_detection_delay()
            )

        active_detected = (
            infector is not None
            and self.get_person_property(infector, "detected")
            and self.bernoulli(self.params["p_active_detect"])
        )

        if active_detected:
            t_active_detected = (
                self.get_person_property(infector, "t_detected")
                + self.generate_active_detection_delay()
            )
            if not detected or t_active_detected < t_detected:
                detected = True
                detect_method = "active"
                t_detected = t_active_detected

        t_recovered = self.get_person_property(id, "t_recovered")
        if detected and t_detected >= t_recovered:
            detected = False
            detect_method = None
            t_detected = None

        return {
            "detected": detected,
            "detect_method": detect_method,
            "t_detected": t_detected,
        }

    def instantiate_secondary_infections(self, id: str) -> list[str]:
        """
        Draw times and instantiate infections for all infections caused by this infection.

        Return their ids.
        """
        secondaries = []

        t_infectious = self.get_person_property(id, "t_infectious")
        if t_infectious is not None:
            generation = self.get_person_property(id, "generation")
            infection_rate = self.get_person_property(id, "infection_rate")
            t_recovered = self.get_person_property(id, "t_recovered")

            secondaries = [
                self.instantiate_infection(
                    id,
                    time + t_infectious,
                    generation + 1,
                )
                for time in self.generate_infection_waiting_times(
                    self.rng,
                    rate=infection_rate,
                    infectious_duration=(t_recovered - t_infectious),
                )
            ]

        self.update_person(id, {"infectees": secondaries})
        return secondaries

    def generate_latent_duration(self) -> float:
        return self.params["latent_duration"]

    def generate_infectious_duration(self) -> float:
        return self.params["infectious_duration"]

    def generate_infection_rate(self) -> float:
        return self.params["infection_rate"]

    def generate_passive_detection_delay(self) -> float:
        return self.params["passive_detection_delay"]

    def generate_active_detection_delay(self) -> float:
        return self.params["active_detection_delay"]

    @staticmethod
    def generate_infection_waiting_times(
        rng: numpy.random.Generator, rate: float, infectious_duration: float
    ) -> np.ndarray:
        """Times from onset of infectiousness to each infection"""
        assert rate >= 0.0
        assert infectious_duration >= 0.0

        if rate == 0.0:
            return np.array(())

        n_events = rng.poisson(infectious_duration * rate)

        # We sort these elsewhere, no need to do extra work
        return rng.uniform(0.0, infectious_duration, n_events)

    def bernoulli(self, p: float) -> bool:
        return self.rng.binomial(n=1, p=p) == 1
