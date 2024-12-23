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


def diamond(xc, yc, width, height, ppl):
    half_x_l = np.linspace(xc - width / 2.0, xc, ppl).tolist()
    half_x_r = np.linspace(xc, xc + width / 2.0, ppl).tolist()
    x = np.array(
        [*half_x_l[:-1], *half_x_r[:-1], *reversed(half_x_r[1:]), *reversed(half_x_l)]
    )  # type: ignore

    half_up_y = np.linspace(yc, yc + height / 2.0, ppl).tolist()
    half_down_y = np.linspace(yc - height / 2.0, yc, ppl).tolist()
    y = np.array(
        [
            *half_up_y[:-1],
            *reversed(half_up_y[1:]),
            *reversed(half_down_y[1:]),
            *half_down_y,
        ]
    )  # type: ignore

    return (
        x,
        y,
    )


def get_end(
    id: str, sim: ringvax.Simulation, stage: str, stage_map: dict[str, dict[str, str]]
):
    if sim.get_person_property(id, "detected") and sim.get_person_property(
        id, "t_detected"
    ) < sim.get_person_property(id, stage_map[stage]["end"]):
        return sim.get_person_property(id, "t_detected")
    else:
        return sim.get_person_property(id, stage_map[stage]["end"])


def mark_detection(ax, id, sim: ringvax.Simulation, plot_par):
    infection = sim.infections[id]
    if sim.get_person_property(id, "detected"):
        y_loc = plot_par["height"][id]
        x_loc = sim.get_person_property(id, "t_detected")
        x_adj = (
            0.5
            * (max(plot_par["x_range"]) - min(plot_par["x_range"]))
            / (max(plot_par["y_range"]) - min(plot_par["y_range"]))
        )
        x, y = diamond(
            x_loc,
            y_loc,
            plot_par["history_thickness"] * x_adj,
            plot_par["history_thickness"],
            plot_par["ppl"],
        )
        ax.fill(x, y, color=plot_par["color"]["detection"][infection["detect_method"]])


def mark_infections(ax, id, sim: ringvax.Simulation, plot_par):
    infection = sim.infections[id]
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


def draw_stages(ax, id, sim: ringvax.Simulation, plot_par):
    infection = sim.infections[id]
    y_loc = plot_par["height"][id]
    for stage in ["latent", "infectious"]:
        x = np.linspace(
            infection[stage_map[stage]["start"]],
            get_end(id, sim, stage, stage_map),
            plot_par["ppl"],
        )

        y = np.array([y_loc] * len(x))

        ax.fill_between(
            x,
            y - plot_par["history_thickness"] / 2.0,
            y + plot_par["history_thickness"] / 2.0,
            color=plot_par["color"][stage],
        )


def connect_child_infections(ax, id, sim: ringvax.Simulation, plot_par):
    y_parent = plot_par["height"][id]
    times_infections = get_infection_time_tuples(id, sim)
    if times_infections is not None:
        for t, inf in times_infections:
            assert (
                sim.infections[inf]["t_exposed"] == t
            ), f"Child {inf} reports infection at time {sim.infections[inf]['t_exposed']} while parent reports time was {t}"
            y_child = plot_par["height"][inf]
            y = np.linspace(
                y_child - plot_par["history_thickness"] / 2.0,
                y_parent - plot_par["history_thickness"] / 2.0,
                plot_par["ppl"],
            )
            x = np.array([t] * len(y))
            ax.plot(x, y, color=plot_par["color"]["connection"])


def get_descendants(id, sim: ringvax.Simulation, offspring: list[str]):
    """
    Get all actually-simulated infections caused by this infection, directly and indirectly.
    """
    infectees = sim.get_person_property(id, "infectees")
    if infectees is not None and len(infectees) > 0:
        for inf in infectees:
            offspring.append(inf)
            get_descendants(sim, inf, offspring)


def get_infection_time_tuples(id: str, sim: ringvax.Simulation):
    infectees = sim.get_person_property(id, "infectees")
    if infectees is None or len(infectees) == 0:
        return None

    return [(sim.get_person_property(inf, "t_exposed"), inf) for inf in infectees]


def order_descendants(sim: ringvax.Simulation):
    # WLOG heights start at 0, the index case, and increase
    # We want to plot the tree as a tree (no crossed lines), which can be done by ensuring that:
    #    1. Each individuals' most recent offspring is plotted at the next height
    #    2. All subsequent offspring heights are adjusted for the number of infections produced by more-recent infections
    order = ["0"]
    _order_descendants("0", sim, order)
    return order


def _order_descendants(id: str, sim: ringvax.Simulation, order: list[str]):
    assert id in order, f"Cannot append infection {id} to ordering list {order}"
    times_infections = get_infection_time_tuples(id, sim)
    print(f"Adding {id} to {order}\n")
    print(f"inf t tups = {times_infections}")
    if times_infections is not None:
        for _, inf in times_infections:
            order.insert(order.index(id) + 1, inf)
            _order_descendants(inf, sim, order)


plot_order = order_descendants(s)

plot_par = {
    "color": {
        "latent": "#888888",
        "infectious": "#c7dcdd",
        "infection": "#888888",
        "connection": "#888888",
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
    "x_range": [
        0.0,
        max(get_end(id, s, "infectious", stage_map) for id in s.infections.keys()),
    ],
    "y_range": [0.0, len(s.infections)],
}

# plot
fig, ax = plt.subplots()


for inf in s.infections.keys():
    draw_stages(ax, inf, s, plot_par)

    mark_detection(ax, inf, s, plot_par)

    mark_infections(ax, inf, s, plot_par)

    connect_child_infections(ax, inf, s, plot_par)

plt.show()
