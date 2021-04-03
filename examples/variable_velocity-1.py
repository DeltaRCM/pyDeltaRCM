def create_velocity_array(end_time, a=1, b=5e4, h=3.2, k=2):
    """Create velocity timeseries.
    """
    _time = np.linspace(0, end_time, num=1000)

    _velocity = a * np.sin((_time - h)/b) + k
    return _time, _velocity