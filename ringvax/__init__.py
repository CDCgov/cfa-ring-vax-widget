import uuid
from typing import Any

import numpy.random


class Simulation:
    def __init__(self, params: dict[str, Any], seed: int = None):
        self.params = params
        self.seed = seed
        self.rng = numpy.random.default_rng(self.seed)
        self.infections = {}

    def run(self):
        self.run_infections()
        self.intervene()

    def create_person(self) -> str:
        """Add a new person to the data"""
        id = str(uuid.uuid4())
        self.infections[id] = {}
        return id

    def get_person_property(self, id: str, property: str) -> Any:
        """Get a property of a person"""
        return self.infections[id][property]

    def query_people(self, query: dict[str, Any]) -> [str]:
        """Get IDs of people with a given set of properties"""
        return [
            id
            for id, person in self.infections.items()
            if all(person[k] == v for k, v in query.items())
        ]

    def update_person(self, id: str, content: dict[str, Any]) -> None:
        self.infections[id] |= content

    def generate_person_properties(
        self, t_exposed: float, infector: str
    ) -> str:
        """Generate properties of a single infected person"""
        # disease state history in this individual, and when they infect others
        infection_history = self.generate_infection_history(
            t_exposed=t_exposed
        )

        # passive detection
        passive_detected = self.binomial(self.params["p_passive_detect"])

        if passive_detected:
            t_passive_detected = (
                t_exposed + self.generate_passive_detection_delay()
            )
        else:
            t_passive_detected = None

        # keep track of generations
        if infector is None:
            generation = 0
        else:
            generation = self.get_person_property(infector, "generation") + 1

        return {
            "infector": infector,
            "generation": generation,
            "passive_detected": passive_detected,
            "t_passive_detected": t_passive_detected,
            "active_detected": None,
            "t_active_detected": None,
            "detected": None,
            "t_detected": None,
            "actually_infected": None,
        } | infection_history

    def run_infections(self) -> None:
        """Run no-intervention counterfactual"""
        # start with the index infection
        index_id = self.create_person()
        self.update_person(
            index_id,
            self.generate_person_properties(t_exposed=0.0, infector=None),
        )

        this_generation = [index_id]

        for generation in range(self.params["n_generations"]):
            next_generation = []

            # instantiate the next-gen infections caused by each infection in this generation
            for infector in this_generation:
                for t_exposed in self.get_person_property(
                    infector, "t_infections"
                ):
                    infectee = self.create_person()
                    self.update_person(
                        infectee,
                        self.generate_person_properties(
                            infector=infector, t_exposed=t_exposed
                        ),
                    )
                    # keep track of the people infected in the next generation
                    next_generation.append(infectee)

            this_generation = next_generation

    def intervene(self) -> None:
        """Draw intervention outcomes and update chains of infection"""
        for generation in range(self.params["n_generations"]):
            # get infections in this generation
            for infectee in self.query_people({"generation": generation}):
                # process each infectee
                self._intervene1(infectee)

    def _intervene1(self, infectee: str) -> None:
        """Process intervention for a single infectee"""
        infector = self.get_person_property(infectee, "infector")

        # did this person actually get infected?
        if infector is None:
            # the index case is always infected
            assert self.get_person_property(infectee, "generation") == 0
            actually_infected = True
        elif not self.get_person_property(infector, "actually_infected"):
            # if the infector was not actually infected, this one also not
            actually_infected = False
        elif self.get_person_property(
            infector, "detected"
        ) and self.get_person_property(
            infector, "t_detected"
        ) < self.get_person_property(
            infectee, "t_exposed"
        ):
            # if the infection occurred after infectee's detection, also not
            # actually infected
            actually_infected = False
        else:
            # otherwise, they are
            actually_infected = True

        # if they were actually infected, see if their infector was detected,
        # so that they have a chance for active detection
        # (infector is not None is to ignore the index infection)
        if (
            actually_infected
            and infector is not None
            and self.get_person_property(infector, "detected")
        ):
            active_detected = self.binomial(self.params["p_active_detect"])

            if active_detected:
                t_active_detected = (
                    self.get_person_property(infector, "t_detected")
                    + self.generate_passive_detection_delay()
                )
            else:
                t_active_detected = None
        else:
            # otherwise, leave these values undefined
            active_detected = None
            t_active_detected = None

        # now reconcile everything that's happened to this person
        # where they detected by either means?
        detected = (
            self.get_person_property(infectee, "passive_detected")
            or active_detected is True
        )
        # if so, when?
        if detected:
            t_passive_detected = self.get_person_property(
                infectee, "t_passive_detected"
            )
            t_detected = min(
                x
                for x in [t_passive_detected, t_active_detected]
                if x is not None
            )

        else:
            t_detected = None

        self.update_person(
            infectee,
            {
                "actually_infected": actually_infected,
                "active_detected": active_detected,
                "t_active_detected": t_active_detected,
                "detected": detected,
                "t_detected": t_detected,
            },
        )

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
            "t_infections": t_infections,
        }

    def generate_latent_duration(self) -> float:
        return self.params["latent_duration"]

    def generate_infectious_duration(self) -> float:
        return self.params["infectious_duration"]

    def generate_infection_rate(self) -> float:
        return self.params["infection_rate"]

    def generate_passive_detection_delay(self) -> float:
        return self.params["passive_detection_delay"]

    @staticmethod
    def generate_infection_delays(
        rng: numpy.random.Generator, rate: float, infectious_duration: float
    ) -> [float]:
        """Times from onset of infectiousness to each infection"""
        assert rate >= 0.0
        assert infectious_duration >= 0.0

        if rate == 0.0:
            return []

        times = []
        # start at t=0, draw the first delay, then add it to the list only if
        # it's inside the infectious duration. then iterate.
        t = rng.exponential(scale=1.0 / rate)
        while t < infectious_duration:
            times.append(t)
            t += rng.exponential(scale=1.0 / rate)

        return times

    def binomial(self, p: float) -> bool:
        return self.rng.binomial(n=1, p=p) == 1
