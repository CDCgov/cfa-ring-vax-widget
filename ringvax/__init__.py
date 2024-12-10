import uuid
from typing import Any

import numpy.random
import scipy.stats


class Simulation:
    def __init__(self, params: dict[str, Any], seed: int = None):
        self.params = params
        self.seed = seed
        self.rng = numpy.random.default_rng(self.seed)

        index_infection = self.get_person()
        self.infections = {index_infection["id"]: index_infection}

    def run(self):
        self.generate_infections()
        self.intervene()

    def generate_infections(self) -> None:
        """Run no-intervention counterfactual"""
        for g in range(self.params["n_generations"]):
            # get list of IDs the infectious people in this generation
            this_generation = [
                id
                for id, person in self.infections.items()
                if person["generation"] == g
            ]

            # instantiate the next-gen infections caused by each infection in this generation
            for infector in this_generation:
                for t_exposed in self.infections[infector]["t_infections"]:
                    infectee = self.generate_person(
                        infector=infector, t_exposed=t_exposed
                    )
                    self.infections.append(infectee)

    def intervene(self) -> None:
        """Draw intervention outcomes and update chains of infection"""
        # figure out who gets detected
        for id in self.infections:
            is_detected_passive = (
                self.rng.uniform() < self.params["p_passive_detect"]
            )

            # do active detection
            pass

        # truncate chains of transmission based on who got detected
        pass

    def generate_person(
        self, t_exposed: float, infector: str = None
    ) -> dict[str, Any]:
        """Generate a single infected person"""
        id = str(uuid.uuid4())
        history = self.get_infection_history(t_exposed=t_exposed)

        if infector is None:
            generation = 0
        else:
            generation = self.infections[infector]["generation"] + 1

        return {
            "id": id,
            "infector": infector,
            "generation": generation,
        } | history

    def get_infection_history(self, t_exposed: float) -> dict[str, Any]:
        """Generate infection history for a single infected person"""
        latent_duration = self.get_latent_duration()
        infectious_duration = self.get_infectious_duration()
        infection_rate = self.get_infection_rate()

        infection_delays = self.get_infection_delays(
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

    def get_latent_duration(self) -> float:
        return self.params["latent_duration"]

    def get_infectious_duration(self) -> float:
        return self.params["infectious_duration"]

    def get_infection_rate(self) -> float:
        return self.params["infection_rate"]

    @staticmethod
    def get_infection_delays(
        rng: numpy.random.Generator, rate: float, infectious_duration: float
    ) -> [float]:
        """Times from onset of infectiousness to each infection"""
        assert rate >= 0.0
        assert infectious_duration >= 0.0

        if rate == 0.0:
            return []

        times = []
        t = scipy.stats.expon.rvs(random_state=rng, scale=1.0)
        while t < infectious_duration:
            times.append(t)
            t += scipy.stats.expon.rvs(random_state=rng, scale=1.0)

        return times
