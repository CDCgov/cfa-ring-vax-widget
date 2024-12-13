from typing import Any, List, Optional

import numpy.random


class Simulation:
    def __init__(self, params: dict[str, Any], seed: Optional[int] = None):
        self.params = params
        self.seed = seed
        self.rng = numpy.random.default_rng(self.seed)
        self.infections = {}

    def create_person(self) -> str:
        """Add a new person to the data"""
        id = str(len(self.infections))
        self.infections[id] = {}
        return id

    def update_person(self, id: str, content: dict[str, Any]) -> None:
        self.infections[id] |= content

    def get_person_property(self, id: str, property: str) -> Any:
        """Get a property of a person"""
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
        infection_queue: List[tuple[float, Optional[str]]] = [(0.0, None)]

        while (
            len(infection_queue) > 0
            and len(self.query_people()) < self.params["max_infections"]
        ):
            # pop a person off the end of the queue
            t_exposed, infector = infection_queue.pop()
            id = self.create_person()
            self.update_person(
                id,
                self.generate_person_properties(t_exposed=t_exposed, infector=infector),
            )

            # process that one person
            self._run1(id)

            # add the infections they caused to the end of the queue
            # unless we are past the number of generations
            if (
                self.get_person_property(id, "generation")
                < self.params["n_generations"]
            ):
                for t in self.get_person_property(id, "t_infections"):
                    # enqueue the next infections, which will be processed unless we hit
                    # the max. # of infections
                    infection_queue.insert(0, (t, id))

    def _run1(self, id: str) -> None:
        """Process a single infection"""
        # run passive detection -------------------------------------------
        passive_detected = self.bernoulli(self.params["p_passive_detect"])

        if passive_detected:
            t_passive_detected = (
                self.get_person_property(id, "t_exposed")
                + self.generate_passive_detection_delay()
            )
        else:
            t_passive_detected = None

        # run active detection --------------------------------------------
        infector = self.get_person_property(id, "infector")

        is_index = infector is None
        if is_index:
            assert self.get_person_property(id, "generation") == 0

        # you are actively detected if you are not the index, and your infector
        # was detected, and you pass the coin flip
        active_detected = (
            not is_index
            and self.get_person_property(infector, "detected")
            and self.bernoulli(self.params["p_active_detect"])
        )

        # if you were actively detected, when?
        if active_detected:
            t_active_detected = (
                self.get_person_property(infector, "t_detected")
                + self.generate_active_detection_delay()
            )
        else:
            t_active_detected = None

        # reconcile detections --------------------------------------------
        detected = active_detected or passive_detected

        if active_detected and passive_detected:
            assert t_passive_detected is not None
            assert t_active_detected is not None
            t_detected = min(t_passive_detected, t_active_detected)
        elif active_detected and not passive_detected:
            assert t_active_detected is not None
            t_detected = t_active_detected
        elif not active_detected and passive_detected:
            assert t_passive_detected is not None
            t_detected = t_passive_detected
        else:
            t_detected = None

        # figure out which infections actually occur ----------------------
        if not detected:
            t_infections = self.get_person_property(id, "t_infections_if_undetected")
        else:
            assert t_detected is not None
            t_infections = [
                t
                for t in self.get_person_property(id, "t_infections_if_undetected")
                if t < t_detected
            ]

        # update my information -------------------------------------------
        self.update_person(
            id,
            {
                "passive_detected": passive_detected,
                "t_passive_detected": t_passive_detected,
                "active_detected": active_detected,
                "t_active_detected": t_active_detected,
                "detected": detected,
                "t_detected": t_detected,
                "t_infections": t_infections,
            },
        )

    def generate_person_properties(
        self, t_exposed: float, infector: Optional[str]
    ) -> dict[str, Any]:
        """Generate properties of a single infected person"""
        # disease state history in this individual, and when they infect others
        infection_history = self.generate_infection_history(t_exposed=t_exposed)

        # keep track of generations
        if infector is None:
            generation = 0
        else:
            generation = self.get_person_property(infector, "generation") + 1

        return {
            "infector": infector,
            "generation": generation,
            "passive_detected": None,
            "t_passive_detected": None,
            "active_detected": None,
            "t_active_detected": None,
            "detected": None,
            "t_detected": None,
            "actually_infected": None,
        } | infection_history

    def generate_infection_history(self, t_exposed: float) -> dict[str, Any]:
        """Generate infection history for a single infected person"""
        latent_duration = self.generate_latent_duration()
        infectious_duration = self.generate_infectious_duration()
        infection_rate = self.generate_infection_rate()

        infection_delays = self.generate_infection_delays(
            self.rng,
            rate=infection_rate,
            infectious_duration=infectious_duration,
        )

        t_infectious = t_exposed + latent_duration
        t_recovered = t_infectious + infectious_duration
        t_infections = [t_infectious + d for d in infection_delays]

        return {
            "t_exposed": t_exposed,
            "t_infectious": t_infectious,
            "t_recovered": t_recovered,
            # N.B.: these are the infections that would occur in the absence of
            # any detection
            "t_infections_if_undetected": t_infections,
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
    def generate_infection_delays(
        rng: numpy.random.Generator, rate: float, infectious_duration: float
    ) -> List[float]:
        """Times from onset of infectiousness to each infection"""
        assert rate >= 0.0
        assert infectious_duration >= 0.0

        if rate == 0.0:
            return []

        n_events = rng.poisson(infectious_duration * rate)
        times = rng.uniform(0.0, infectious_duration, n_events)
        times.sort()
        return list(times)

    def bernoulli(self, p: float) -> bool:
        return self.rng.binomial(n=1, p=p) == 1
