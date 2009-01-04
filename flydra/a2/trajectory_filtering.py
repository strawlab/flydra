import warnings

def iterate_over_subtrajectories(style=None,
                                 min_samples=None,
                                 data=None,
                                 stimulus=None):
    """break data into subtrajectories according to style

    Sample usage:

    for rows in iterate_over_subtrajectories(style='not on walls',
        min_samples=10,
        data=all_rows,
        stimulus=stimulus):
    """
    warnings.warn('subtrajectories not yet implemented')
    yield data
