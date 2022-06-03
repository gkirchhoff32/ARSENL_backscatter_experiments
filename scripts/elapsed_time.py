def get_t_elap(max_rollover, rollover_num, final_event_tstamp):
    """
    Return total time elapsed from INPHAMIS dataset.
    Args:
        max_rollover: The maximum timestamp before instrument timer rolls over.
        rollover_num: Number of rollover events.
        final_event_tstamp: Time stamp of final event the follows the last rollover in time.

    Returns:
        t_elap: Total time elapsed

    """

    t_elap = rollover_num * max_rollover + final_event_tstamp  # total time elapsed

    return t_elap
