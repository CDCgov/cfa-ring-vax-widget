import matplotlib.pyplot as plt
import numpy as np

from ringvax import Simulation

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
    )

    return (
        x,
        y,
    )


def half_hourglass(upper, xc, yc, width, height, ppl):
    k = 1.0 if upper else -1.0
    x = np.array(
        [
            *(np.linspace(xc, xc + width / 2.0, ppl).tolist())[:-1],
            *(np.linspace(xc + width / 2.0, xc - width / 2.0, ppl).tolist())[:-1],
            *(np.linspace(xc - width / 2.0, xc, ppl).tolist())[:-1],
        ]
    )  # type: ignore
    y = np.array(
        [
            *(np.linspace(yc, yc + k * height / 2.0, ppl).tolist()[:-1]),
            *([yc + k * height / 2.0] * (ppl - 1)),
            *(np.linspace(yc + k * height / 2.0, yc, ppl).tolist()[:-1]),
        ]
    )

    return (
        x,
        y,
    )


def get_end(id: str, sim: Simulation, stage: str, stage_map: dict[str, dict[str, str]]):
    """
    Get time of end of given stage for given infection.

    Returns t_detected if this is less than natural end time of the stage, even if the stage did not start.
    """
    if sim.get_person_property(id, "detected") and sim.get_person_property(
        id, "t_detected"
    ) < sim.get_person_property(id, stage_map[stage]["end"]):
        return sim.get_person_property(id, "t_detected")
    else:
        return sim.get_person_property(id, stage_map[stage]["end"])


def mark_detection(ax, id, sim: Simulation, plot_par):
    """
    If this infection was detected, mark that.
    """
    if sim.get_person_property(id, "detected"):
        y_loc = plot_par["height"][id]
        x_loc = sim.get_person_property(id, "t_detected")
        x_adj = (
            0.5
            * (max(plot_par["x_range"]) - min(plot_par["x_range"]))
            / (max(plot_par["y_range"]) - min(plot_par["y_range"]))
        )
        dtype = sim.get_person_property(id, "detect_method")
        if dtype == "active":
            x, y = diamond(
                x_loc,
                y_loc,
                plot_par["history_thickness"] * x_adj,
                plot_par["history_thickness"],
                plot_par["ppl"],
            )
            ax.fill(
                x,
                y,
                color=plot_par["color"]["detection"],
            )
        elif dtype == "passive":
            x, y = half_hourglass(
                True,
                x_loc,
                y_loc,
                plot_par["history_thickness"] * x_adj,
                plot_par["history_thickness"],
                plot_par["ppl"],
            )
            ax.fill(
                x,
                y,
                color=plot_par["color"]["detection"],
            )
            x, y = half_hourglass(
                False,
                x_loc,
                y_loc,
                plot_par["history_thickness"] * x_adj,
                plot_par["history_thickness"],
                plot_par["ppl"],
            )
            ax.fill(
                x,
                y,
                color=plot_par["color"]["detection"],
            )
        else:
            raise RuntimeError("Unknown detection type {dtype}")


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
        x = np.linspace(
            sim.get_person_property(id, stage_map[stage]["start"]),
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
                y_parent - plot_par["history_thickness"] / 2.0,
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


def make_plot_par(sim: Simulation):
    """
    Get parameters for plotting this simulation
    """
    plot_order = order_descendants(sim)

    return {
        "color": {
            "latent": "#c7dcdd90",
            "infectious": "#cf482890",
            "infection": "#888888",
            "connection": "#888888",
            "detection": "#20419a",
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
            max(
                get_end(id, sim, "infectious", stage_map)
                for id in sim.infections.keys()
            ),
        ],
        "y_range": [0.0, len(sim.infections)],
    }


def plot_simulation(sim: Simulation):
    plot_par = make_plot_par(sim)

    fig, ax = plt.subplots()

    for inf in sim.query_people():
        draw_stages(ax, inf, sim, plot_par)

        mark_detection(ax, inf, sim, plot_par)

        mark_infections(ax, inf, sim, plot_par)

        connect_child_infections(ax, inf, sim, plot_par)

    ax.set_axis_off()
    return fig
