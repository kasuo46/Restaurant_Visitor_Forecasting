from sklearn.metrics import mean_squared_log_error


def rmsle(y, pred):
    return mean_squared_log_error(y, pred)**0.5
