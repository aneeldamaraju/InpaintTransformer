import numpy as np

def get_pts_on_curve(xy,P=50):
    '''
    Given the radius of a curve, output evenly spaced points along the curve
    Output dimension = 1XPX2
    '''
    xlist = xy[0,::int(xy.shape[1]/P)][:int(P),0]
    ylist = xy[0,::int(xy.shape[1]/P)][:int(P),1]

    pts = np.array([[int(x), int(y)] for (x, y) in zip(xlist, ylist)])
    pts = pts.reshape((1, -1, 2))
    return pts
