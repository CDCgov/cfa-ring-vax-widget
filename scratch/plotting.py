import matplotlib.pyplot as plt
import numpy as np

import ringvax

params = {
    "n_generations": 6,
    "latent_duration": 1.0,
    "infectious_duration": 3.0,
    "infection_rate": 0.6,
    "p_passive_detect": 0.5,
    "passive_detection_delay": 2.0,
    "p_active_detect": 0.15,
    "active_detection_delay": 2.0,
    "max_infections": 1000000,
}

s = ringvax.Simulation(params=params, rng=np.random.default_rng(1))
s.run()

len(s.infections)


stage_map = {
    "latent": {"start": "t_exposed", "end": "t_infectious"},
    "infectious": {"start": "t_infectious", "end": "t_recovered"},
}


def get_end(infection, stage, stage_map):
    if (
        infection["detected"]
        and infection["t_detected"] < infection[stage_map[stage]["end"]]
    ):
        return infection["t_detected"]
    else:
        return infection[stage_map[stage]["end"]]


def mark_detection(ax, id, infections, plot_par):
    infection = infections[id]
    y_loc = plot_par["height"][id]
    if infection["detected"]:
        y = np.linspace(
            y_loc - plot_par["history_thickness"] / 2.0,
            y_loc + plot_par["history_thickness"] / 2.0,
            plot_par["ppl"],
        )
        x = np.array([infection["t_detected"]] * len(y))
        ax.plot(x, y, color=plot_par["color"]["detection"][infection["detect_method"]])


def mark_infections(ax, id, infections, plot_par):
    infection = infections[id]
    y_loc = plot_par["height"][id]
    if infection["infection_times"].size > 0:
        for t in infection["infection_times"]:
            y = np.linspace(
                y_loc - plot_par["history_thickness"] / 2.0,
                y_loc + plot_par["history_thickness"] / 2.0,
                plot_par["ppl"],
            )
            x = np.array([t] * len(y))
            ax.plot(x, y, color=plot_par["color"]["infection"])


def draw_stages(ax, id, infections, plot_par):
    infection = infections[id]
    y_loc = plot_par["height"][id]
    for stage in ["latent", "infectious"]:
        x = np.linspace(
            infection[stage_map[stage]["start"]],
            get_end(infection, stage, stage_map),
            plot_par["ppl"],
        )

        y = np.array([y_loc] * len(x))

        ax.fill_between(
            x,
            y - plot_par["history_thickness"] / 2.0,
            y + plot_par["history_thickness"] / 2.0,
            color=plot_par["color"][stage],
        )


def connect_child_infections(ax, id, infections, plot_par):
    print(f"Connecting children of {id}")
    infection = infections[id]
    y_parent = plot_par["height"][id]
    infectees = infection["infectees"]
    if infectees is not None and len(infectees) > 0:
        times = infection["infection_times"]
        infectees = infection["infectees"]
        for neg_t, inf in sorted(zip(-times, infectees)):
            assert (
                infections[inf]["t_exposed"] == -neg_t
            ), f"Child {inf} reports infection at time {infections[inf]['t_exposed']} while parent reports time was {-neg_t}"
            y_child = plot_par["height"][inf]
            y = np.linspace(
                y_child - plot_par["history_thickness"] / 2.0,
                y_parent - plot_par["history_thickness"] / 2.0,
                plot_par["ppl"],
            )
            x = np.array([-neg_t] * len(y))
            ax.plot(x, y, color=plot_par["color"]["connection"])


def get_descendants(sim: ringvax.Simulation, infection: str, offspring: list[str]):
    """
    Get all actually-simulated infections caused by this infection, directly and indirectly.
    """
    infectees = sim.get_person_property(infection, "infectees")
    if infectees is not None and len(infectees) > 0:
        for inf in infectees:
            offspring.append(inf)
            get_descendants(sim, inf, offspring)


def get_infection_time_tuples(sim: ringvax.Simulation, id: str):
    infectees = sim.get_person_property(id, "infectees")
    if infectees is None or len(infectees) == 0:
        return None

    times_infections = [
        (sim.get_person_property(inf, "t_exposed"), inf) for inf in infectees
    ]
    return times_infections


def order_descendants(sim: ringvax.Simulation):
    # WLOG heights start at 0, the index case, and increase
    # We want to plot the tree as a tree (no crossed lines), which can be done by ensuring that:
    #    1. Each individuals' most recent offspring is plotted at the next height
    #    2. All subsequent offspring heights are adjusted for the number of infections produced by more-recent infections
    order = ["0"]
    _order_descendants(sim, "0", order)
    return order


def _order_descendants(sim: ringvax.Simulation, id: str, order: list[str]):
    assert id in order, f"Cannot append infection {id} to ordering list {order}"
    times_infections = get_infection_time_tuples(sim, id)
    if times_infections is not None:
        for _, inf in times_infections:
            order.insert(order.index(id) + 1, inf)
            _order_descendants(sim, inf, order)


plot_order = order_descendants(s)

plot_par = {
    "color": {
        "latent": "#c7dcdd",
        "infectious": "#cf4828",
        "infection": "#000000",
        "connection": "#bbbbbb",
        "detection": {
            "active": "#068482",
            "passive": "#f78f47",
        },
    },
    "history_thickness": 0.5,
    "linewidth": {
        "connection": 2.0,
        "detection": 10.0,
        "infection": 10.0,
    },
    "ppl": 100,
    "height": {
        inf: len(plot_order) - 1.0 * height for height, inf in enumerate(plot_order)
    },
}

# plot
fig, ax = plt.subplots()


for inf in s.infections.keys():
    draw_stages(ax, inf, s.infections, plot_par)

    mark_detection(ax, inf, s.infections, plot_par)

    mark_infections(ax, inf, s.infections, plot_par)

    connect_child_infections(ax, inf, s.infections, plot_par)

plt.show()
