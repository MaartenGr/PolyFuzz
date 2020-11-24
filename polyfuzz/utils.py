import logging


def check_matches(model):
    """ Checks if matches were created by verifying the presence of self.matches

    Arguments:
        model: polyfuzz instance for which the check is performed.

    Returns:
        None

    Raises:
        NotFittedError: If the matches were not found.
    """
    msg = ("This %(name)s instance is not fitted yet. Call 'match' with "
           "appropriate arguments before using this estimator.")

    if not model.matches:
        raise ValueError(msg % {'name': type(model).__name__})


def check_grouped(model):
    """ Checks if matches were created by verifying the presence of self.matches

    Arguments:
        model: polyfuzz instance for which the check is performed.

    Returns:
        None

    Raises:
        NotFittedError: If the matches were not found.
    """
    msg = ("This %(name)s instance is not grouped yet. Call 'group' with "
           "appropriate arguments before using this estimator.")

    if not model.cluster_mappings and not model.clusters:
        raise ValueError(msg % {'name': type(model).__name__})


def create_logger():
    """ Initialize logger """
    logger = logging.getLogger('BERTopic')
    logger.setLevel(logging.WARNING)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
    logger.addHandler(sh)
    return logger
