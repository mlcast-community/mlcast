import numpy as np
import pysteps


def forecast(precip: np.ndarray, velocity: np.ndarray, timesteps: float | list[float], **keywords):
    """precip: array-like
    Array of shape (ar_order+1,m,n) containing the input precipitation fields
    ordered by timestamp from oldest to newest. The time steps between
    the inputs are assumed to be regular.
        velocity: array-like
    Array of shape (2,m,n) containing the x- and y-components of the
    advection field.
    The velocities are assumed to represent one time step between the
    inputs. All values are required to be finite.
        timesteps: int or list of floats
    Number of time steps to forecast or a list of time steps for which the
    forecasts are computed (relative to the input time step). The elements
    of the list are required to be in ascending order.
        ...
        Returns
        -------
        out: ndarray
    A three-dimensional array of shape (num_timesteps,m,n) containing a time
    series of forecast precipitation fields. The time series starts from
    t0+timestep, where timestep is taken from the input precipitation fields
    precip. If measure_time is True, the return value is a three-element
    tuple containing the nowcast array, the initialization time of the
    nowcast generator and the time used in the main loop (seconds).
    """


# example use

# Load example with precipitation in mm/h
precipitation, metadata, timestep = pysteps.datasets.load_dataset("mrms", frames=35)

# subset in time (for flow estimation) and transform
train_precip = precipitation[0:5]
train_precip_dbr, metadata_dbr = pysteps.utils.transformation.dB_transform(
    train_precip, metadata, threshold=0.1, zerovalue=-15.0
)

# Import the Lucas-Kanade optical flow algorithm
oflow_method = pysteps.motion.get_method("LK")

# Estimate the motion field from the training data (in dBR)
motion_field = oflow_method(train_precip_dbr)

# use all but first two first frames as reference for nowcast
observed_precip = precipitation[3:]
n_leadtimes = observed_precip.shape[0]
precip_forecast = pysteps.nowcasts.extrapolate(train_precip[-1], motion_field, n_leadtimes)
