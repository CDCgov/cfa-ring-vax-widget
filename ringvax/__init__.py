import bisect
from typing import Any, List, Optional

import numpy as np
import numpy.random


class Simulation:
    PROPERTIES = {
        "id",
        "infector",
        "infectees",
        "generation",
        "t_exposed",
        "t_infectious",
        "t_recovered",
        "infection_rate",
        "detected",
        "detect_method",
        "t_detected",
        "infection_times",
    }

    def __init__(
        self, params: dict[str, Any], rng: Optional[numpy.random.Generator] = None
    ):
        self.params = params
        self.rng = rng if rng is not None else numpy.random.default_rng()
        self.infections = {}
        self.termination: Optional[str] = None

    def create_person(self) -> str:
        """Add a new person to the data"""
        id = str(len(self.infections))
        self.infections[id] = {x: None for x in self.PROPERTIES}
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

    def register_infectee(self, infector, infectee) -> None:
        infectees = self.get_person_property(infector, "infectees")
        if infectees is None:
            self.update_person(infector, {"infectees": []})
            infectees = self.get_person_property(infector, "infectees")
        infectees.append(infectee)

    def run(self) -> None:
        """Run simulation"""
        # queue is pairs (t_exposed, infector)
        # start with the index infection
        infection_queue: List[tuple[float, Optional[str]]] = [(0.0, None)]

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

            # find the person who is infected next
            # (the queue is time-sorted, so this is the temporally next infection)
            t_exposed, infector = infection_queue.pop(0)

            # otherwise, instantiate this infection, draw who they in turn infect,
            # and add the infections they cause to the queue, in time order
            id = self.create_person()
            self.generate_infection(id=id, t_exposed=t_exposed, infector=infector)

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
                for t in self.get_person_property(id, "infection_times"):
                    bisect.insort_right(infection_queue, (t, id), key=lambda x: x[0])

    def generate_infection(
        self, id: str, t_exposed: float, infector: Optional[str]
    ) -> None:
        """
        Generate a single infected person's biological disease history, detection
        history and transmission history
        """
        # keep track of generations
        if infector is None:
            generation = 0
        else:
            generation = self.get_person_property(infector, "generation") + 1
            self.register_infectee(infector, id)

        self.update_person(
            id, {"id": id, "infector": infector, "generation": generation}
        )

        # disease state history in this individual
        disease_history = self.generate_disease_history(t_exposed=t_exposed)
        self.update_person(id, disease_history)

        # whether this person was detected
        detection_history = self.generate_detection_history(id)
        self.update_person(id, detection_history)

        t_start_infectious = disease_history["t_infectious"]
        if detection_history["detected"]:
            t_end_infectious = detection_history["t_detected"]
        else:
            t_end_infectious = disease_history["t_recovered"]

        # when do they infect people?
        infection_rate = self.generate_infection_rate()

        if disease_history["t_infectious"] > t_end_infectious:
            infection_times = np.array([])
        else:
            infection_times = (
                t_start_infectious
                + self.generate_infection_waiting_times(
                    self.rng,
                    rate=infection_rate,
                    infectious_duration=(
                        t_end_infectious - disease_history["t_infectious"]
                    ),
                )
            )
            assert (infection_times >= t_start_infectious).all()
            assert (infection_times <= t_end_infectious).all()

        self.update_person(id, {"infection_times": infection_times})

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
