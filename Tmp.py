from sklearn.neighbors import DistanceMetric
from numpy import fliplr
from numpy import pi

def _unscaled_dist(self, X, X2=None):
    """
    Compute the haversine distance between each row of X and X2, or between
    each pair of rows of X if X2 is None. First column must be longitude and
    the second latitude, both in degrees.
    """
    haversine = DistanceMetric.get_metric('haversine')
    if X2 is None:
        return haversine.pairwise(fliplr(X)*pi/180.)
        # note sklearn haversine distance requires (lat, long) whereas we are
        # working with (long, lat). numpy.fliplr switches the columns of X.
    else:
        return 6371*haversine.pairwise(fliplr(X)*pi/180., fliplr(X2)*pi/180.)

def unscaled_dist(X, X2=None):
    """
    Compute the haversine distance between each row of X and X2, or between
    each pair of rows of X if X2 is None. First column must be longitude and
    the second latitude,both in degrees.
    """
    haversine = DistanceMetric.get_metric('haversine')
    if X2 is None:
        return 6371*haversine.pairwise(fliplr(X)*pi/180.)
        # note sklearn haversine distance requires (lat, long) whereas we are
        # working with (long, lat). numpy.fliplr switches the columns of X.
    else:
        return 6371*haversine.pairwise(fliplr(X)*pi/180., fliplr(X2)*pi/180.)
