
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nra


def dead_simple_1d_filter_step(t, x, state_dict, m = 1.0):
    # (t, x) are the new input point
    # t should probably increment by 1 each time for typical application
    # state_dict contains old floats {mass, slope, t_bar, t2_bar, x_bar, x2_bar}
    # you can set m to something other than 1.0 to weight points differently
    # returns new state_dict

    # Unpack old parameters
    M = float(state_dict["mass"])
    a = float(state_dict["slope"])
    t_bar = float(state_dict["t_bar"])
    t2_bar = float(state_dict["t2_bar"])
    x_bar = float(state_dict["x_bar"])
    x2_bar = float(state_dict["x2_bar"])

    # get timescale
    sigma_t2 = t2_bar - t_bar * t_bar
    sigma_x2 = x2_bar - x_bar * x_bar
    lam = (sigma_x2 - a**2 * sigma_t2) / sigma_x2
    M_old = lam * M
    M_new = m + M_old

    if lam < 0 or lam > 1:
        raise ValueError("current lambda = {}".format(lam), " should probably be 0 <= lambda <= 1")

    # calculate slope
    I = M_old * sigma_t2
    I_delta =  m * M_old / M_new * (t - t_bar) ** 2
    a_delta = (x - x_bar) / (t - t_bar)
    a_new = (a * I + a_delta * I_delta) / (I + I_delta)

    # update parameters
    t_bar_new = (m * t + M_old * t_bar) / M_new
    t2_bar_new = (m * t**2 + M_old * t2_bar) / M_new
    x_bar_new = (m * x + M_old * x_bar) / M_new
    x2_bar_new = (m * x**2 + M_old * x2_bar) / M_new

    new_state = {
        "mass": M_new,
        "slope": a_new,
        "t_bar": t_bar_new,
        "t2_bar": t2_bar_new,
        "x_bar": x_bar_new,
        "x2_bar": x2_bar_new,
    }

    return new_state