from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ringvax import Simulation

stage_map = {
    "latent": {"start": "t_exposed", "end": "t_infectious"},
    "infectious": {"start": "t_infectious", "end": "t_recovered"},
}


def fwd_triangle(x0, yc, width, height, ppl):
    x = np.array(
        [
            *([x0] * (ppl - 1)),
            *(np.linspace(x0, x0 + width, ppl).tolist())[:-1],
            *(np.linspace(x0 + width, x0, ppl).tolist())[:-1],
        ]
    )  # type: ignore
    y = np.array(
        [
            *(np.linspace(yc - height / 2.0, yc + height / 2.0, ppl).tolist()[:-1]),
            *(np.linspace(yc + height / 2.0, yc, ppl).tolist()[:-1]),
            *(np.linspace(yc, yc - height / 2.0, ppl).tolist()[:-1]),
        ]
    )

    return (
        x,
        y,
    )


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


def get_stage_info(
    id: str, sim: Simulation, stage: str, stage_map: dict[str, dict[str, str]]
):
    """
    Get real and counterfactual duration of this stage.

    Returns a tuple, with [0] the real history portion and [1] the counterfactual
        [0] the [0] start and [1] end times of the part of this stage that actually happened (both None if it was pre-empted)
        [1] the [0] start and [1] end times of the part of this stage that was counterfactual (both None if no detection)
    """
    t_start = sim.get_person_property(id, stage_map[stage]["start"])
    t_end = sim.get_person_property(id, stage_map[stage]["end"])
    detected = sim.get_person_property(id, "detected")
    t_detected = None if not detected else sim.get_person_property(id, "t_detected")
    if detected and t_detected < t_start:
        return (
            (None, None),
            (
                t_start,
                t_end,
            ),
        )
    elif sim.get_person_property(id, "detected") and t_detected < t_end:
        return (
            (t_start, t_detected),
            (t_detected, t_end),
        )
    else:
        return (
            (
                t_start,
                t_end,
            ),
            (None, None),
        )


def mark_detection(ax, id, sim: Simulation, plot_par, fixed_height: bool = False):
    """
    If this infection was detected, mark that.
    """
    if (
        sim.get_person_property(id, "detected")
        and sim.get_person_property(id, "detect_method") == "active"
    ):
        y_loc = plot_par["height"][id]
        x_loc = sim.get_person_property(id, "t_detected")
        y_adj = 1.5
        x_adj = y_adj / 2.0
        if fixed_height:
            x_adj = (
                2
                / 3
                * (max(plot_par["x_range"]) - min(plot_par["x_range"]))
                / (max(plot_par["y_range"]) - min(plot_par["y_range"]))
            )
        x, y = fwd_triangle(
            x_loc,
            y_loc,
            plot_par["history_thickness"] * x_adj,
            plot_par["history_thickness"] * y_adj,
            plot_par["ppl"],
        )
        ax.fill(
            x,
            y,
            color=plot_par["color"]["detection"],
        )


def mark_infections(ax, id, sim: Simulation, plot_par):
    """
    Put down tick marks at every time a new infection arises caused by given infection.
    """
    y_loc = plot_par["height"][id]
    if (sim.get_person_property(id, "infection_times")).size > 0:
        for t in sim.get_person_property(id, "infection_times"):
            y = np.linspace(
                y_loc - plot_par["history_thickness"] / 2.0,
                y_loc + plot_par["history_thickness"] / 2.0,
                plot_par["ppl"],
            )
            x = np.array([t] * len(y))
            ax.plot(x, y, color=plot_par["color"]["infection"])


def draw_stages(ax, id, sim: Simulation, plot_par):
    """
    Draw the stages (latent, infectious) of this infection
    """
    y_loc = plot_par["height"][id]
    for stage in ["latent", "infectious"]:
        info = get_stage_info(id, sim, stage, stage_map)
        # The part of the stage that actually happened
        if info[0][0] is not None:
            start, end = info[0]
            assert start is not None and end is not None
            x = np.linspace(
                start,
                end,
                plot_par["ppl"],
            )

            y = np.array([y_loc] * len(x))

            ax.fill_between(
                x,
                y - plot_par["history_thickness"] / 2.0,
                y + plot_par["history_thickness"] / 2.0,
                alpha=plot_par["alpha"]["real"],
                color=plot_par["color"][stage],
            )
        # Counterfactual, if detection happened
        if plot_par["show_counterfactual"] and info[1][0] is not None:
            start, end = info[1]
            assert start is not None and end is not None

            x = np.linspace(
                start,
                end,
                plot_par["ppl"],
            )

            y = np.array([y_loc] * len(x))

            ax.fill_between(
                x,
                y - plot_par["history_thickness"] / 2.0,
                y + plot_par["history_thickness"] / 2.0,
                alpha=plot_par["alpha"]["counterfactual"],
                color=plot_par["color"][stage],
            )

    if plot_par["annotate_generation"]:
        x_loc = max(x)
        gen = sim.get_person_property(id, "generation")
        plt.text(x_loc, y_loc, f" ({gen})", va="center", size="small")


def connect_child_infections(ax, id, sim: Simulation, plot_par):
    """
    Connect this infection to its children
    """
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
                y_parent + plot_par["history_thickness"] / 2.0,
                plot_par["ppl"],
            )
            x = np.array([t] * len(y))
            ax.plot(x, y, color=plot_par["color"]["connection"])


def get_infection_time_tuples(id: str, sim: Simulation):
    """
    Get tuple of (time, id) for all infections this infection causes.
    """
    infectees = sim.get_person_property(id, "infectees")
    if infectees is None or len(infectees) == 0:
        return None

    return [(sim.get_person_property(inf, "t_exposed"), inf) for inf in infectees]


def order_descendants(sim: Simulation):
    """
    Get infections in order for plotting such that the tree has no crossing lines.

    We order such that, allowing space for all descendants thereof, the most recent
    infection caused by any infection is closest to it.
    """
    order = ["0"]
    _order_descendants("0", sim, order)
    return order


def _order_descendants(id: str, sim: Simulation, order: list[str]):
    """
    Add this infections descendants in order
    """
    assert id in order, f"Cannot append infection {id} to ordering list {order}"
    times_infections = get_infection_time_tuples(id, sim)
    if times_infections is not None:
        for _, inf in times_infections:
            order.insert(order.index(id) + 1, inf)
            _order_descendants(inf, sim, order)


def make_plot_par(sim: Simulation, show_counterfactual=True):
    """
    Get parameters for plotting this simulation
    """
    plot_order = order_descendants(sim)

    return {
        "annotate_generation": True,
        "show_counterfactual": show_counterfactual,
        "color": {
            "latent": "#4477AA",
            "infectious": "#EE6677",
            "infection": "#000000",
            "connection": "#000000",
            "detection": "#000000",
        },
        "alpha": {
            "real": 1.0,
            "counterfactual": 0.25,
        },
        "history_thickness": 0.25,
        "ppl": 100,
        "height": {
            inf: len(plot_order) - 1.0 * height - 1.0
            for height, inf in enumerate(plot_order)
        },
        "x_range": [
            0.0,
            max(
                sim.get_person_property(id, stage_map["infectious"]["end"])
                for id in sim.infections.keys()
            ),
        ],
        "y_range": [-1.0, len(sim.infections)],
    }


def plot_simulation(sim: Simulation, par: dict[str, Any]):
    n_inf = len(sim.query_people())

    plot_par = make_plot_par(sim) | par

    fig, ax = plt.subplots()

    for inf in sim.query_people():
        draw_stages(ax, inf, sim, plot_par)

        mark_infections(ax, inf, sim, plot_par)

        mark_detection(ax, inf, sim, plot_par)

        connect_child_infections(ax, inf, sim, plot_par)

    if n_inf == 1:
        plot_par["history_thickness"] = 1.5 * plot_par["history_thickness"]
    ax.set(ylim=plot_par["y_range"])
    ax.yaxis.set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_label_position("bottom")
    ax.set_xlabel("time")
    fig.set_figheight(0.2 * n_inf)
    return fig
