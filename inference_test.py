from __future__ import print_function
import math
import sys
import h5py as h5
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

def logprior_line(pars):
    return np.zeros(pars.shape[:-1])

def logpost_line(pars, X, Y, Yerr):
    return logprior_line(pars) + loglike_line(pars,X,Y,Yerr)

def logprior_hyper(pars):
    if (pars > 10.0).any():
        return -np.inf
    return 0.0

def loglike_hyper(pars, samples, edges):
    ndim = edges.shape[0]
    nb = edges.shape[1]-1
    pars2 = np.exp(np.reshape(pars, (ndim,nb)))

    vals = pars2[-1]
    for i in xrange(ndim-2,-1,-1):
        par = pars2[i]
        ns = np.ones(ndim-i)
        ns[0] = nb
        vals = np.tile(vals, ns) * par
    ll = - (pars2*(edges[:,1:]-edges[:,:-1])).sum(axis=1).prod()
    for i, chain in enumerate(samples):
        res = np.histogramdd(chain, edges)
        p = res[0] * vals

        ll += math.log(np.sum(p))
    
    return ll

def logpost_hyper(pars, samples, edges):
    lprior = logprior_hyper(pars)
    if lprior == -np.inf:
        return -np.inf
    return logprior_hyper(pars) + loglike_hyper(pars, samples, edges)

def sample_post(logpost, args, guess, nwalkers=100, ndim=2, burn=100, run=100):
    p0 = [guess*np.random.normal(1.0,0.01,ndim) for i in xrange(nwalkers)]
    sampler = em.EnsembleSampler(nwalkers, ndim, logpost, args=args)
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

def plot_hist(edges, counts, norm=1.0, *args, **kwargs):
    x = np.array(list(zip(edges[:-1],edges[1:]))).flatten()
    bh = counts / np.diff(edges)
    y = np.array(list(zip(bh,bh))).flatten()
    plt.plot(x, y, *args, **kwargs)

if __name__ == "__main__":

    try:
        n = int(sys.argv[1])
    except:
        n = 3

    nwalkers = 50
    burn = 200
    run = 200
    ndim = 2
    thin = 200
    nb = 4
    maxm =  5.0
    minm = -5.0
    maxb =  5.0
    minb = -5.0

    m = np.random.normal(size=n)
    b = np.random.normal(size=n)
    m[m>maxm] = maxm
    m[m<minm] = minm
    b[b>maxb] = maxb
    b[b<minb] = minb

    chains = np.zeros((n, nwalkers*(run/thin), ndim))
    edges = np.zeros((ndim, nb+1))
    edges[0] = np.linspace(minm, maxm, nb+1)
    edges[1] = np.linspace(minb, maxb, nb+1)
    guess = np.zeros((ndim, nb))
    guess[0] = 1.0/(maxm-minm)
    guess[1] = 1.0/(maxb-minb)
    guess = np.log(guess)

    for i in xrange(n):
        print("{0:d} of {1:d}".format(i+1, n))
        X, Y, Yerr = make_line_data(m[i], b[i], 0.2, 20, 0.0, 5.0)
        chain, logprob = sample_post(logpost_line, [X,Y,Yerr], [0.1,0.1], 
                nwalkers, ndim, burn, run)
    #    plot_fit(chain, logprob, X, Y, Yerr, m, b, thin)
        chains[i,:,:] = chain[:,::thin,:].reshape(-1,ndim)
       
    print(chains.shape)

    hyper_burn = 1
    hyper_run = 1000
    hyper_thin = 50
    hyper_keep = 500

    chain, logprob = sample_post(logpost_hyper, [chains,edges], 
            guess.flatten(), nb*nwalkers, ndim*nb, hyper_burn, hyper_run)

    #fig = tri.corner(chain.reshape(-1,ndim*nb))

    print(chain.shape)
    print(edges.shape)

    plt.figure()
    for i in xrange(nb*nwalkers):
        for j in xrange(hyper_keep,hyper_run,hyper_thin):
            plt.subplot(2,1,1)
            plot_hist(edges[0], np.exp(chain[i,j,:nb]), 1.0, color='k', alpha=0.05)
            plt.subplot(2,1,2)
            plot_hist(edges[1], np.exp(chain[i,j,nb:]), 1.0, color='k', alpha=0.05)
    
    plt.subplot(2,1,1)
    plt.hist(m, bins=nb, range=[minm,maxm], color='r', histtype='step')
    plt.xlabel(r"$m$")
    plt.subplot(2,1,2)
    plt.hist(b, bins=nb, range=[minb,maxb], color='r', histtype='step')
    plt.ylabel(r"$b$")
    
    plt.show()
