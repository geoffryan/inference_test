from __future__ import print_function
import math
import numpy as np
import emcee
import matplotlib.pyplot as plt
import sys

## For data gen
def genLinear(n,a,b):
    return (b-a) * np.sqrt(np.random.rand(n)) + a

def genNormal(n,mu,sig):
    return np.random.normal(mu, sig, n)

def genLineData(m, b, sig, A, B, n):
    N = m.shape[0]
    X = (B-A) * np.random.rand(N, n) + A
    Y = m[:,None] * X[:,:] + b[:,None] + np.random.normal(0.0,sig,size=(N,n))
    err = sig * np.ones(X.shape)
    return X, Y, err

## For MCMC
def logprob(theta, X, Y, err):
    m = theta[0]
    b = theta[1]
    res = (Y - (m*X+b)) / err
    return -0.5 * (res*res).sum()

def hyperlogprob(theta,samples,mbins,bbins):

    if (theta<0.0).any():
        return -np.inf

    nm = mbins.shape[0]-1
    nb = bbins.shape[0]-1

    theta2d = theta.reshape(nm,nb)
    dm = mbins[1:]-mbins[:-1]
    db = bbins[1:]-bbins[:-1]
    J = (theta2d * dm[:,None]*db[None,:]).sum()

    bins = (mbins,bbins)

    logprob = 0.0
    for sample in samples:
        N = sample.shape[0]
        counts, medge, bedge = np.histogram2d(sample[:,0], sample[:,1], bins)
        logprob += math.log((counts*theta2d).sum()/N)

    logprob -= J

    return logprob

def genSamples(X, Y, err, M, B, xA, xB):

    n = X.shape[0]

    nburn = 200
    nsteps = 400
    nwalkers = 20

    # samples has shape [nobj, nsamples]
    samples = np.zeros((n,nwalkers*(nsteps-nburn),2))
    lnprobability = np.zeros((n,nwalkers*(nsteps-nburn)))
    map = np.zeros((n,2))

    for i in range(n):
        initial = [np.random.random(2) for j in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, 2, logprob, 
                                        args=(X[i],Y[i],err[i]))

        print("Sampling {0:d} of {1:d}...".format(i+1,n))

        j = 0
        for r in sampler.sample(initial, iterations=nsteps):
            print("  {0:.1f}%".format((100.0*j)/nsteps), end='\r')
            sys.stdout.flush()
            j += 1
        print("  {0:.1f}%".format((100.0*j)/nsteps), end='\r')
        sys.stdout.flush()

        samples[i,:,:] = sampler.chain[:,nburn:,:].reshape((-1,2))
        lnprobability[i,:] = sampler.lnprobability[:,nburn:].reshape(-1)

        fig, ax = plt.subplots(3)
        XX = np.linspace(xA, xB, 100)
        YY = M[i]*XX + B[i]
        best = sampler.flatchain[sampler.lnprobability.argmax()]
        map[i,:] = best
        YYY = best[0]*XX + best[1]
        ax[0].plot(XX, YY, lw=2.0, color='r', ls='-')
        ax[0].errorbar(X[i], Y[i], err[i], ls='', color='k')
        ax[0].plot(XX, YYY, lw=2.0, ls='-', color='b')
        ax[1].axhline(M[i], color='r', lw=2.0)
        ax[1].axhline(best[0], color='b', lw=2.0)
        ax[2].axhline(B[i], color='r', lw=2.0)
        ax[2].axhline(best[1], color='b', lw=2.0)
        for k in xrange(nwalkers):
            YY = sampler.chain[k,-1,0]*XX + sampler.chain[k,-1,1]
            ax[0].plot(XX, YY, lw=1.0, color='k', ls='-', alpha=0.2)
            ax[1].plot(sampler.chain[k,:,0], color='k', ls='-', alpha=0.2)
            ax[2].plot(sampler.chain[k,:,1], color='k', ls='-', alpha=0.2)
        ax[0].set_xlabel(r"$x$")
        ax[0].set_ylabel(r"$y$")
        ax[0].set_xlim(xA, xB)
        ax[1].set_ylabel(r"$m$")
        ax[2].set_xlabel(r"step")
        ax[2].set_ylabel(r"$b$")
        plt.tight_layout()
        fig.savefig("samples_{0:03d}.png".format(i))
        plt.close()

    return samples, lnprobability, map

def genHyperSamples(samples, mbins, bbins, prefix):
    
    nburn = 0
    nsteps = 1000
    nwalkers = 200

    n = samples.shape[0]
    ndim = (mbins.shape[0]-1) * (bbins.shape[0]-1)

    initial = np.array([np.random.random(ndim) for i in range(nwalkers)])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, hyperlogprob, 
                                    args=(samples,mbins,bbins))

    print("Sampling samples!")

    j = 0
    for r in sampler.sample(initial, iterations=nsteps):
        print("  {0:.1f}%".format((100.0*j)/nsteps), end='\r')
        sys.stdout.flush()
        j += 1
    print("  100.0%", end='\r')
    sys.stdout.flush()

    return sampler.chain, sampler.lnprobability

def myHist(ax, data, nb, a, b, norm, *args, **kwargs):
    bins = np.linspace(a, b, nb+1)
    counts, bins1 = np.histogram(data, bins)
    X = np.zeros(2*(nb+1))
    Y = np.zeros(2*(nb+1))
    X[::2] = bins
    X[1::2] = bins
    Y[1:-1:2] = counts / (bins[1:]-bins[:-1]) * (float(norm)/(len(data.flat)))
    Y[2::2] = counts / (bins[1:]-bins[:-1]) * (float(norm)/(len(data.flat)))
    Y[0] = 0.0
    Y[-1] = 0.0

    ax.plot(X, Y, *args, **kwargs)

# Python's main

if __name__ == "__main__":

    # Generate data
    n = 50   # number of objects
    m = 10  # number of points
    thin = 100
    sig = 4.0
    mMU = 0.0
    mSIG = 1.0
    bA = -5.0
    bB = 5.0
    xA = 0.0
    xB = 10.0
    M = genNormal(n, mMU, mSIG)
    B = genLinear(n, bA, bB)
    X, Y, err = genLineData(M, B, sig, xA, xB, m)

    # Run MCMC
    samples, lnprob, map = genSamples(X, Y, err, M, B, xA, xB)
    
    nb = 10
    NB = 4

    mA = mMU - 4*mSIG
    mB = mMU + 4*mSIG

    mbins = np.linspace(mA, mB, NB+1)
    bbins = np.linspace(bA, bB, NB+1)

    hypersamples, hyperlnprobability = genHyperSamples(samples[:,::thin,], 
                                                    mbins, bbins, "samples")
    #hypersamples = genHyperSamples(mus,nbins, lb, rb, mus, "mus")

    fig, ax = plt.subplots(2)

    XX = np.linspace(mA, mB, 200)
    ax[0].plot(XX, n * np.exp(-0.5*((XX-mMU)/mSIG)**2)
            / math.sqrt(2*math.pi*mSIG*mSIG), lw=2, color='r')
    myHist(ax[0], M, nb, mA, mB, n, lw=2.0, color='k')
    myHist(ax[0], map[:,0], nb, mA, mB, n, 'b')
    myHist(ax[0], samples[:,:,0], 4*nb, mA, mB, n, 'k')
    ax[0].set_xlabel(r"$m$")
    ax[0].set_ylabel(r"$dN/dm$")
    
    XX = np.linspace(bA, bB, 200)
    ax[1].plot(XX, n * 2*(XX-bA)/(bB-bA)**2, lw=2, color='r')
    myHist(ax[1], B, nb, bA, bB, n, lw=2.0, color='k')
    myHist(ax[1], map[:,1], nb, bA, bB, n, 'b')
    myHist(ax[1], samples[:,:,1], 4*nb, bA, bB, n, 'k')
    ax[1].set_xlabel(r"$b$")
    ax[1].set_ylabel(r"$dN/db$")

    for theta in hypersamples[:,-1]:
        theta2d = theta.reshape(NB,NB)
        mrates = theta2d.sum(axis=1)
        brates = theta2d.sum(axis=0)
        XX = np.zeros(2*(NB+1))
        YY = np.zeros(2*(NB+1))
        XX[::2] = mbins
        XX[1::2] = mbins
        YY[1:-2:2] = mrates
        YY[2:-1:2] = mrates
        YY[0] = 0.0
        YY[-1] = 0.0
        ax[0].plot(XX, YY, color='g', alpha = 0.1)
        XX[::2] = bbins
        XX[1::2] = bbins
        YY[1:-2:2] = brates
        YY[2:-1:2] = brates
        YY[0] = 0.0
        YY[-1] = 0.0
        ax[1].plot(XX, YY, color='g', alpha = 0.1)

    plt.tight_layout()
    fig.savefig("hist.png")
    plt.close()

    fig, ax = plt.subplots()
    for k in xrange(hyperlnprobability.shape[0]):
        ax.plot(hyperlnprobability[k], 'k', alpha=0.1)
    fig.savefig("hyperlogprob.png")
    plt.close()


    #nbins = 4
    #lb = -5.0
    #rb = 15.0

    #for i in xrange(n):
    #    print("Plotting histogram: {0:d}".format(i))
    #    compare_point_post(samples[i], data[i], sig[i], mus[i], 0.0, b, str(i))


