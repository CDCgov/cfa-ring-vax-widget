import bisect
from typing import Any, List, Optional

import numpy as np
import numpy.random


class Simulation:
    INIT_PROPERTIES = {
        "id",
        "infector",
        "generation",
        "t_exposed",
        "simulated",
    }
    SIM_PROPERTIES = {
        "infectees",
        "t_infectious",
        "t_recovered",
        "infection_rate",
        "detected",
        "detect_method",
        "t_detected",
        "infection_times",
    }
    PROPERTIES = INIT_PROPERTIES | SIM_PROPERTIES

    def __init__(
        self, params: dict[str, Any], rng: Optional[numpy.random.Generator] = None
    ):
        self.params = params
        self.rng = rng if rng is not None else numpy.random.default_rng()
        self.infections = {}
        self.termination: Optional[str] = None

    def create_person(
        self, infector: Optional[str], t_exposed: float, generation: int
    ) -> str:
        """Add a new person to the data"""
        id = str(len(self.infections))
        self.infections[id] = {
            "id": id,
            "infector": infector,
            "t_exposed": t_exposed,
            "generation": generation,
            "simulated": False,
        } | {x: None for x in self.SIM_PROPERTIES}
        return id

    def update_person(self, id: str, content: dict[str, Any]) -> None:
        bad_properties = set(content.keys()) - set(self.PROPERTIES)
        if len(bad_properties) > 0:
            raise RuntimeError(f"Properties not in schema: {bad_properties}")

        self.infections[id] |= content

    def get_person_property(self, id: str, property: str) -> Any:
        """Get a property of a person"""
        if property not in self.PROPERTIES:
            raise RuntimeError(f"Property '{property}' not in schema")

        if id not in self.infections:
            raise RuntimeError(
                f"No person with {id=}; cannot get property '{property}'"
            )
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
        # queue is of infection ids
        # start with the index infection
        infection_queue: List[str] = [self.create_person(None, 0.0, 0)]

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
            elif n_infections >= self.params["max_infections"]:
                # we are at maximum number of infections
                raise RuntimeError("Maximum number of infections exceeded")

            # find the person who is infected next
            # (the queue is time-sorted, so this is the temporally next infection)
            id = infection_queue.pop(0)

            # draw who they in turn infect,
            # and add the infections they cause to the queue, in time order
            offspring = self.generate_infection(id=id)

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
                for child in offspring:
                    bisect.insort_right(
                        infection_queue,
                        child,
                        key=lambda x: self.get_person_property(x, "t_exposed"),
                    )

    def generate_infection(
        self,
        id: str,
    ) -> List[str]:
        """
        Generate a single infected person's biological disease history, detection
        history and transmission history
        """

        # disease state history in this individual
        disease_history = self.generate_disease_history(
            self.get_person_property(id, "t_exposed")
        )
        self.update_person(id, disease_history)

        # whether this person was detected
        detection_history = self.generate_detection_history(id)
        self.update_person(id, detection_history)

        t_end_infectious = (
            detection_history["t_detected"]
            if detection_history["detected"]
            else disease_history["t_recovered"]
        )

        # when do they infect people?
        infection_rate = self.generate_infection_rate()

        if disease_history["t_infectious"] > t_end_infectious:
            infection_times = np.array([])
        else:
            infection_times = disease_history[
                "t_infectious"
            ] + self.generate_infection_waiting_times(
                self.rng,
                rate=infection_rate,
                infectious_duration=(
                    t_end_infectious - disease_history["t_infectious"]
                ),
            )
            assert (infection_times >= disease_history["t_infectious"]).all()
            assert (infection_times <= t_end_infectious).all()

        infectees = [
            self.create_person(id, time, self.get_person_property(id, "generation") + 1)
            for time in infection_times
        ]
        self.update_person(
            id, {"infection_times": infection_times, "infectees": infectees}
        )

        # mark this person as simulated
        self.update_person(id, {"simulated": True})

        return infectees

    def generate_disease_history(self, t_exposed: float) -> dict[str, Any]:
        """Generate infection history for a single infected person"""
        latent_duration = self.generate_latent_duration()
        infectious_duration = self.generate_infectious_duration()
        infection_rate = self.generate_infection_rate()

        t_infectious = t_exposed + latent_duration
        t_recovered = t_infectious + infectious_duration

        return {
            "t_exposed": t_exposed,
            "t_infectious": t_infectious,
            "t_recovered": t_recovered,
            "infection_rate": infection_rate,
        }

    def generate_detection_history(self, id: str) -> dict[str, Any]:
        """Determine if a person is infected, and when"""
        # determine properties of the infector
        infector = self.get_person_property(id, "infector")

        infector_detected = infector is not None and self.get_person_property(
            infector, "detected"
        )

        t_infector_detected = (
            self.get_person_property(infector, "t_detected")
            if infector_detected
            else None
        )

        # determine what kinds of detection this individual is eligible for
        potentially_passive_detected = self.bernoulli(self.params["p_passive_detect"])

        potentially_active_detected = infector_detected and self.bernoulli(
            self.params["p_active_detect"]
        )

        # actually determine what kind of detection occurred, if any
        return self.resolve_detection_history(
            potentially_passive_detected=potentially_passive_detected,
            potentially_active_detected=potentially_active_detected,
            passive_detection_delay=self.generate_passive_detection_delay(),
            active_detection_delay=self.generate_active_detection_delay(),
            t_exposed=self.get_person_property(id, "t_exposed"),
            t_recovered=self.get_person_property(id, "t_recovered"),
            t_infector_detected=t_infector_detected,
        )

    @staticmethod
    def resolve_detection_history(
        potentially_passive_detected: bool,
        potentially_active_detected: bool,
        passive_detection_delay: float,
        active_detection_delay: float,
        t_exposed: float,
        t_recovered: float,
        t_infector_detected: Optional[float],
    ) -> dict[str, Any]:
        # a "detection" is a tuple (time, method)
        detections = []

        # keep track of passive and active possibilities
        if potentially_passive_detected:
            detections.append((t_exposed + passive_detection_delay, "passive"))

        if potentially_active_detected:
            assert t_infector_detected is not None
            detections.append((t_infector_detected + active_detection_delay, "active"))

        # detection only actually happens if it's before recovery
        detections = [x for x in detections if x[0] < t_recovered]

        if len(detections) == 0:
            return {"detected": False, "t_detected": None, "detect_method": None}
        else:
            # choose the earliest detection
            detection = min(detections, key=lambda x: x[0])
            return {
                "detected": True,
                "t_detected": detection[0],
                "detect_method": detection[1],
            }

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
        rng: numpy.random.Generator,
        rate: float,
        infectious_duration: float,
    ) -> np.ndarray:
        """Times from onset of infectiousness to each infection"""
        assert rate >= 0.0
        assert infectious_duration >= 0.0

        if rate == 0.0:
            return np.array(())

        n_events = rng.poisson(infectious_duration * rate)

        # We sort these elsewhere
        return rng.uniform(0.0, infectious_duration, n_events)

    def bernoulli(self, p: float) -> bool:
        return self.rng.binomial(n=1, p=p) == 1
