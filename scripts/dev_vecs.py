
import random

import itertools

import numpy as np
from genereg import draw

def get_unit_random_vectors(num_vecs, dim, scale = None, dist = "normal", normalize = True):
    
    if scale is None:
        scale = 1/dim
    
    if dist == "normal":
        vecs = np.random.normal(0, scale, size = (num_vecs, dim))
    elif dist == "uniform":
        vecs = scale*(2*np.random.random(size = (num_vecs, dim))-1)
    
    if normalize:
        vecs = vecs/np.sqrt(np.sum(np.pow(vecs, 2), axis = 1))[:,None]
    
    return vecs

def sample_ips_combi(num_vecs, dim, num_samples, scale = None, dist = "normal", normalize = True):
    
    v1s = get_unit_random_vectors(num_vecs, dim, scale=scale, dist=dist, normalize = normalize)
    isamps = np.random.randint(0, num_vecs, size = num_samples)
    jsamps = np.random.randint(0, num_vecs, size = num_samples)
    ips = np.vecdot(v1s[isamps], v1s[jsamps])
    return ips

def sample_ips_rand(num_vecs, dim, num_samples, scale = None, dist = "normal", normalize = True):
    
    v1s = get_unit_random_vectors(num_samples, dim, scale=scale, dist=dist, normalize = normalize)
    v2s = get_unit_random_vectors(num_samples, dim, scale=scale, dist=dist, normalize = normalize)
    ips = np.vecdot(v1s, v2s)
    return ips

def test_random_vecs(num_vecs, dim, num_samples = None, mode = "rand", scale = None, dist = "normal", normalize = True):
    
    if num_samples is None:
        num_samples = num_vecs * (num_vecs - 1) // 2
    elif isinstance(num_samples, float):
        num_samples = int(num_samples * num_vecs * (num_vecs - 1) // 2)
    
    if mode == "rand":
        ips = sample_ips_rand(num_vecs, dim, num_samples, scale=scale, dist=dist, normalize = normalize)
    elif mode == "combi":
        ips = sample_ips_combi(num_vecs, dim, num_samples, scale=scale, dist=dist, normalize = normalize)
    
    mean = np.mean(ips)
    sd = np.std(ips)
    minn = min(ips)
    maxx = max(ips)
    
    print(f"Results for {num_vecs} {dist} distributed random vectors with dimension {dim}, {num_samples} samples ({mode} mode):")
    print(f"inner product mean: {mean:0.3f}, sd {sd:0.3f}, min {minn:0.3f}, max {maxx:0.3f}")
    
    hist, bins = np.histogram(ips, bins = max(10, int(np.sqrt(num_vecs))), range=(-1,1), density = True)
    
    sctxt = draw.scalar_to_text_nb(hist, minval = 0, add_range= True)
    print("inner products:")
    for r in sctxt:
        print(r)
    print()    
    
    ip2s = np.power(ips, 2)
    hist, bins = np.histogram(ip2s, bins = max(10, int(np.sqrt(num_vecs))), range=(0,1), density = True)
    
    sctxt = draw.scalar_to_text_nb(hist, minval = 0, add_range= True)
    sctxt, _ = draw.add_ruler(sctxt, xmin = 0, xmax = 1, num_labels = 5, ticks = 0, minor_ticks = 2, fstr = "0.2f")
    print("inner products^2:")
    for r in sctxt:
        print(r)
    print()    
    
    return ips

def vary_dim(num_vecs, min_dim, max_dim, num_samples = None, mode = "rand", scale = None, dist = "normal", normalize = True):
    
    for d in range(min_dim, max_dim+1):
        print(f"***** dim = {d} *****")
        test_random_vecs(num_vecs, d, num_samples = num_samples, mode = mode, scale=scale, dist=dist, normalize = normalize)
        print()

def main():
    
    num_rounds = 3
    num_vecs = 1000
    dim = 3
    min_dim = 2
    max_dim = 4
    num_samples = 100000
    
    scale = None
    # dist = "normal"
    dist = "uniform"
    
    # normalize = True
    normalize = False
    
    # vary_dim(num_vecs, min_dim, max_dim, num_samples = num_samples, mode = "rand", scale = scale, dist = dist)
    vary_dim(num_vecs, min_dim, max_dim, num_samples = num_samples, mode = "combi", scale = scale, dist = dist, normalize=normalize)
    



if __name__=="__main__":
    main()



