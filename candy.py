import numpy as np
from collections import deque
from scipy.optimize import leastsq
from scipy.interpolate import UnivariateSpline
from scipy.stats import chisquare, linregress

def evaluate_groups_and_write(groups,fn):
    """ Calculates some metrics and saves results as a recarray.
    - Envelope participation: higher values should mean a cleaner group
    - DM-Sigma curve fitting: chisquares for Gaussian and Lorentzian profiles
    - Width-Sigma correlation: should be close to -1 (computed 2 ways)
    - Time-DM fitting: R**2 for a linear regression, should be close to 1 (computed 2 ways)
    """
    ev = []
    for g in groups:
        g = g.view(np.recarray)
        if g.sigma.max() < 8: continue 
        ev.append(evaluate_group(g))
    ev = np.array(ev)
    fields = "env dsG dsL ws1 ws2 td1 td2"
    dt = [(f,np.float64) for f in fields.split()]
    ev = ev.view(dt).view(np.recarray)
    np.save(fn, ev)

def evaluate_group(group):
    return (envelope_participation(group),
            dm_sigma_shape(group)[0],
            dm_sigma_shape(group)[1],
            sigma_width_correlation(group),
            sigma_width_correlation(group,best=False),
            dm_time_behaviour(group),
            dm_time_behaviour(group,use_env=False))


### GROUP TESTS ###

def envelope_participation(group):
    """ Fraction of group that's a part of the envelope """
    env = get_envelope(group)
    return float(len(env))/len(group)

def dm_sigma_shape(group):
    """ Chi-squares for Gaussian and Lorentzian profiles """
    env = get_envelope(group)
    i_xs, i_ys = interp_envelope(env)
    G_fit = fit_gauss(i_xs, i_ys)
    L_fit = fit_lorentz(i_xs, i_ys)
    e_xs, e_ys = env.dm, env.sigma
    g_ys, l_ys = G_fit(e_xs), L_fit(e_xs)
    return (chisquare(e_ys, g_ys, 2)[0], chisquare(e_ys, l_ys, 2)[0])

def sigma_width_correlation(group,best=True):
    """ Check for negative correlation between snr and boxcar width 
    -  best=True  : uses the highest snr for each boxcar
    -  best=False : uses all points 
    """ 
    env = get_envelope(group)
    env.sort(order='sigma')
    if best:
        env = env[::-1]
        sub = env[np.unique(env.downfact,True)[1]]
        return np.corrcoef(sub.downfact,sub.sigma)[0,1]
    else:
        return np.corrcoef(env.downfact,env.sigma)[0,1]

def dm_time_behaviour(group, use_env=True):
    """ Checks for straightness in dm-time 
    (compound groups should fit poorly) """
    pts = get_envelope(group) if use_env else group
    _,_,R,_,_ = linregress(pts.time, pts.dm)
    return R**2


### ENVELOPE THINGS ###

def get_envelope(group, k=5):
    """ Ascending maxima with window size k """
    group.sort(order='dm')
    X = np.vstack([np.arange(len(group)),group.sigma]).T
    deq = deque()
    env = set()
    for x in X:
        while len(deq) and deq[-1][1] <= x[1]: deq.pop()
        deq.append(x)
        while deq[0][0] <= x[0]-k: deq.popleft()
        env.add(deq[0][0])
    return group[sorted(list(env))]

def interp_envelope(env):
    """ Generates a spline interpolation from envelope points """
    spl = UnivariateSpline(env.dm, env.sigma, k=5, s=10)
    dms = np.linspace(env.dm[0], env.dm[-1], 256)
    return dms, spl(dms)


### CURVE FITTING ###

def fit(func, p0, xs, ys):
    errfunc = lambda p, x, y: func(p,x) - y
    p1, _   = leastsq(errfunc, p0[:], args=(xs,ys))
    return lambda x: func(list(p1), x)

def fit_gauss(xs, ys):
    func = lambda p, x: p[0]*np.exp(-((x-p[1])/p[2])**2)+5.0
    p0   = [ys.max(), xs[np.argmax(ys)], np.sqrt(ys.var())]
    return fit(func, p0, xs, ys)

def fit_lorentz(xs, ys):
    func = lambda p, x: p[0]/((x-p[1])**2+p[2]**2)+5.0
    p0   = [ys.max(), xs[np.argmax(ys)], np.sqrt(ys.var())]
    return fit(func, p0, xs, ys)    
