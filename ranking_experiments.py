import numpy as np
from collections import deque
from scipy.optimize import leastsq
from scipy.interpolate import UnivariateSpline
from scipy.stats import chisquare, linregress
from RRATrap import SinglePulseGroup

def run_tests(groups=None,fn='candy_out'):
    """ Calculates some experimental metrics and saves results as a recarray.
    - Envelope participation: higher values should mean a cleaner group
    - DM-Sigma curve fitting: chisquares for Gaussian and Lorentzian profiles
    - Width-Sigma correlation: should be close to -1 (computed 2 ways)
    - Time-DM fitting: R**2 for a linear regression, should be close to 1 (computed 2 ways)
    """
    if groups is None: groups = read_groups()
    ev = np.array([group_tests(g) for g in groups])
    fields = 'time dm snr rank env dsG dsL ws1 ws2 td1 td2'
    dt = [(f,np.float64) for f in fields.split()]
    ev = ev.view(dt).view(np.recarray)
    np.save(fn, ev)

def group_tests(group):
    sps = group.singlepulses.view(np.recarray)
    return (sps.time[np.argmax(sps.sigma)],
            sps.dm[np.argmax(sps.sigma)],
            sps.sigma.max(),
            group.rank,
            envelope_participation(sps,3),
            dm_sigma_shape(sps)[0],
            dm_sigma_shape(sps)[1],
            sigma_width_correlation(sps),
            sigma_width_correlation(sps,best=False),
            dm_time_behaviour(sps),
            dm_time_behaviour(sps,use_env=False))


### READING EXISTING RRATRAP OUTPUT ###
def gobble(f, n):
    for _ in xrange(n): next(f)

def read_groups(fn='groups.txt'):
    groups = []
    dtype=np.dtype([('dm', 'float32'),
                    ('sigma','float32'),
                    ('time','float32'),
                    ('sample','uint32'),
                    ('downfact','uint8')])
    with open(fn, 'r') as f:
        gobble(f,8)
        for l in f:
            if 'Group' in l: 
                gobble(f,5)
                rank = int(float(next(f).split()[-1]))
                gobble(f,2)
                group = []
            elif len(l.split()) == 0: 
                new_group = SinglePulseGroup(np.array(group).reshape((len(group),)).view(np.recarray))
                new_group.rank = rank
                groups.append(new_group)
            else: 
                group.append(np.array([tuple(l.split())], dtype=dtype))
    return groups


### GROUP TESTS ###
def envelope_participation(sps,k=5):
    """ Fraction of group that's a part of the envelope """
    env = get_envelope(sps,k)
    return float(len(env))/len(sps)

def dm_sigma_shape(sps):
    """ Chi-squares for Gaussian and Lorentzian profiles """
    env = get_envelope(sps)
    i_xs, i_ys = interp_envelope(env)
    G_fit = fit_gauss(i_xs, i_ys)
    L_fit = fit_lorentz(i_xs, i_ys)
    e_xs, e_ys = env.dm, env.sigma
    g_ys, l_ys = G_fit(e_xs), L_fit(e_xs)
    return (chisquare(e_ys, g_ys, 2)[0], chisquare(e_ys, l_ys, 2)[0])

def sigma_width_correlation(sps,best=True):
    """ Check for negative correlation between snr and boxcar width 
    -  best=True  : uses the highest snr for each boxcar
    -  best=False : uses all points 
    """ 
    env = get_envelope(sps)
    env.sort(order='sigma')
    if best:
        env = env[::-1]
        sub = env[np.unique(env.downfact,True)[1]]
        return np.corrcoef(sub.downfact,sub.sigma)[0,1]
    else:
        return np.corrcoef(env.downfact,env.sigma)[0,1]

def dm_time_behaviour(sps, use_env=True):
    """ Checks for straightness in dm-time 
    (compound groups should fit poorly) """
    pts = get_envelope(sps) if use_env else sps
    _,_,R,_,_ = linregress(pts.time, pts.dm)
    return R**2


### ENVELOPE THINGS ###
def get_envelope(sps, k=5):
    """ Ascending maxima with window size k """
    sps.sort(order='dm')
    X = np.vstack([np.arange(len(sps)),sps.sigma]).T
    deq = deque()
    env = set()
    for x in X:
        while len(deq) and deq[-1][1] <= x[1]: deq.pop()
        deq.append(x)
        while deq[0][0] <= x[0]-k: deq.popleft()
        env.add(deq[0][0])
    return sps[sorted(list(env))]

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

###############################################################

if __name__ == "__main__": run_tests()
