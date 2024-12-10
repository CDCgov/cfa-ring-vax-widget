from typing import Any

import numpy.random
import scipy.stats


class Simulation:
    def __init__(self, params: dict[str, Any], seed: int = None):
        self.params = params
        self.seed = seed
        self.rng = numpy.random.default_rng(self.seed)

        index_infection = self.get_infection_history(t_exposed=0.0) | {"generation": 0}
        self.infections = [index_infection]

    def run(self):
        for g in range(self.params["n_generations"]):
            # get all the infectious people in this generation
            this_generation = [x for x in self.infections if x["generation"] == g]

            # instantiate the next-gen infections caused by each infection in this generation
            for x in this_generation:
                for t_exposed in x["t_infections"]:
                    self.infections.append(
                        self.get_infection_history(t_exposed) | {"generation": g + 1}
                    )

    def get_infection_history(self, t_exposed: float) -> dict[str, Any]:
        latent_duration = self.get_latent_duration()
        infectious_duration = self.get_infectious_duration()
        infection_rate = self.get_infection_rate()

        infection_delays = self.get_infection_delays(
            self.rng, rate=infection_rate, infectious_duration=infectious_duration
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
