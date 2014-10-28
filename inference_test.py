from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
import emcee as em
import triangle as tri

def make_line_data(m, b, sig, nx, x0, x1):
    X = x0 + (x1-x0)*np.random.rand(nx)
    X.sort()
    Yerr = np.random.normal(scale=sig, size=nx)
    Y = m*X + b + Yerr
    return X, Y, sig*np.ones(nx)

def loglike_line(pars, X, Y, Yerr):
    dy = (pars[0]*X+pars[1] - Y) / Yerr
    return -0.5*np.sum(dy*dy)

def logprior_line(pars, X, Y, Yerr):
    return 0.0

def logpost_line(pars, X, Y, Yerr):
    return logprior_line(pars,X,Y,Yerr) + loglike_line(pars,X,Y,Yerr)

def sample_line(X, Y, Yerr, nwalkers=100, ndim=2, burn=100, run=100):
    p0 = [0.05*np.random.rand(ndim) for i in xrange(nwalkers)]
    sampler = em.EnsembleSampler(nwalkers, ndim, logpost_line, args=[X,Y,Yerr])
    i=0
    print("Burning in...")
    for result in sampler.sample(p0, iterations=burn, storechain=False):
        print("{0:.1f}%".format(100.0*i/burn), end='\r')
        sys.stdout.flush()
        i += 1
    p1 = result[0]
    prob1 = result[1]
    state1 = result[2]
    sampler.reset()
    
    i=0
    print("Sampling...")
    for result in sampler.sample(p1, iterations=run, storechain=True):
        print("{0:.1f}%".format(100.0*i/run), end='\r')
        sys.stdout.flush()
        i += 1
    print(sampler.acor)

    return sampler.chain, sampler.lnprobability

def plot_fit(chain, logprob, X, Y, Yerr, m, b, thin=10):
    nwalkers = chain.shape[0]
    run = chain.shape[1]
    ndim = chain.shape[2]

    fig1 = plt.figure()
    plt.errorbar(X, Y, Yerr, fmt='k.')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    for j in xrange(nwalkers):
        for k in xrange(0,run,thin):
            p = chain[j,k]
            plt.plot(X, p[0]*X+p[1], color='k', alpha = 0.05)
    plt.plot(X, m[i]*X+b[i], 'r')
    samples = np.zeros((nwalkers, run, ndim+1))
    samples[:,:,:ndim] = chain
    samples[:,:,-1] = np.tile(np.arange(run),(nwalkers,1))
    labels = [r"$m$", r"$b$", "#"]
    fig2 = tri.corner(samples.reshape(run*nwalkers,ndim+1), truths=[m[i],b[i],0], labels=labels)

if __name__ == "__main__":

    try:
        n = int(sys.argv[1])
    except:
        n = 3

    m = np.random.normal(size=n)
    b = np.random.normal(size=n)

    nwalkers = 100
    burn = 100
    run = 200
    ndim = 2
    thin = 40

    for i in xrange(n):
        print("{0:d} of {1:d}".format(i+1, n))
        X, Y, Yerr = make_line_data(m[i], b[i], 0.2, 20, 0.0, 5.0)
        chain, logprob = sample_line(X, Y, Yerr, nwalkers, ndim, burn, run)
        plot_fit(chain, logprob, X, Y, Yerr, m, b, thin)
        
plt.show()
