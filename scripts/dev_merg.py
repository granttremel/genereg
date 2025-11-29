

import json
import numpy as np
from datetime import datetime
import os

from tabulate import tabulate

from scipy.stats import describe

from genereg.network import gene, genome
from genereg.network.gene import Gene, Allele, PhenotypicGene, PhenotypicAllele

from genereg.network.genome import Genome

from genereg import utils, organism
from genereg.organism import Organism, Population

def get_genome(num_genes, num_phenotypes, density, **kwargs):
    
    mean_init_product = kwargs.pop("mean_init_product", 0.1)
    sd_init_product = kwargs.pop("sd_init_product", 0.0)
    
    gnm = Genome.initialize_random(num_genes, num_phenotypes, density, **kwargs)
    
    gnm.initialize(mean_init_product = mean_init_product, sd_init_product = sd_init_product)
    
    return gnm

def try_genomes(num_genomes, t_max, num_genes, num_phenotypes, density, **kwargs):
    
    orgs = []
    
    rsrc = kwargs.get("resource", -1)
    
    # for n in range(num_genomes):
        
    while True:
        orgs = []
        for ng in range(num_genomes):
            gnm = get_genome(num_genes, num_phenotypes, density, **kwargs)
            org = Organism(gnm, resource = rsrc)
            org.step_to(t_max)
            
            orgs.append(org)
        orgs[0].show_genome()
        Organism.plot_expressions(*orgs, genes = False, phenos = True, headers = [f"Genome{i}" for i in range(num_genomes)])
        
        res = input("keep genome?")
        if res:
            
            try:
                ni = int(res)
            except:
                continue
            
            return orgs[ni]
    
    # Organism.plot_interesting_genes(*orgs, topk = 5, headers = [f"Genome{i}" for i in range(num_genomes)])
    # print("^^^^^^^^^^^^^ interesting ^^^^^^^^^^^^^^^^^^^^^^^")
    Organism.plot_expressions(*orgs, genes = False, phenos = False, headers = [f"Genome{i}" for i in range(num_genomes)])
    
    return orgs

def make_merged_genome(gnm1, gnm2, num_bridges, density, **kwargs):
    return Genome.merge_genomes([gnm1, gnm2], num_bridges, tags = ["A","B"], density = density, **kwargs)

def try_merged_genomes(num_genomes, t_max, num_genes, num_phenotypes, density, num_bridges, bridge_density, resource, **kwargs):
    
    mean_init_product = kwargs.pop("mean_init_product", 0.1)
    sd_init_product = kwargs.pop("sd_init_product", 0.0)
    
    while True:
        
        orgs = []
        for n in range(num_genomes):
            gnm1 = get_genome(num_genes, num_phenotypes, density, **kwargs)
            gnm2 = get_genome(num_genes, num_phenotypes, density, **kwargs)
            
            mgnm = make_merged_genome(gnm1, gnm2, num_bridges, bridge_density, **kwargs)
            org = Organism(mgnm, resource = resource)
            org.initialize(mean_init_product = mean_init_product, sd_init_product = sd_init_product)
            org.step_to(t_max)
            orgs.append(org)
        
        orgs[0].show_genome()
        Organism.plot_expressions(*orgs, genes = False, headers = [f"Indi{n}" for n in range(len(orgs))])
        res = input("keep genome?")
        if res:
            
            try:
                ni = int(res)
            except:
                continue
            
            return orgs[ni]

def make_individuals(gnm:Genome, num_indis, rate, p, resource, **kwargs):
    
    t_max = kwargs.get("t_max", 20)
    mean_init_product = kwargs.pop("mean_init_product", 0.1)
    sd_init_product = kwargs.pop("sd_init_product", 0.0)
    
    org = Organism(gnm, resource = -1)
    org.initialize(mean_init_product = mean_init_product, sd_init_product = sd_init_product)
    org.step_to(t_max)
    inds = [org]
    
    for i in range(num_indis - 1):
        
        new_gnm = gnm.with_mutant_alleles(rate, p)
        new_org = Organism(new_gnm, resource = resource)
        inds.append(new_org)
        new_org.initialize(mean_init_product = mean_init_product, sd_init_product = sd_init_product)
        new_org.step_to(t_max)
        
        gnm = new_gnm
        
    print("Base Genome:")
    org.show_genome()
    Organism.plot_expressions(*inds, genes = False, headers = [f"Indi{n}" for n in range(len(inds))])
    
    return inds

def try_individuals(gnm:Genome, num_indis, rate, p, resource, **kwargs):
    
    while True:
        indis = make_individuals(gnm, num_indis, rate, p, resource, **kwargs)
        
        res = input("keep genome?")
        if res:
            
            try:
                ni = int(res)
            except:
                continue
            
            return indis[ni]


def try_generation(gnm:Genome, pop_size, num_timesteps, mutation_rate, mutation_p, resource, **kwargs):
    
    mean_offs = kwargs.pop("mean_offs", 2)
    mean_parity = kwargs.pop("mean_parity", 2)
    epoch = kwargs.pop("epoch", 20)
    
    pop = Population(gnm)
    
    new_gnm = gnm
    for n in range(pop_size):
        org = Organism(new_gnm, resource = resource)
        pop.add_individual(org)
        gnm = new_gnm
        new_gnm = gnm.with_mutant_alleles(mutation_rate, mutation_p)
    
    pop.individuals[0].show_genome()
    indis = []
    
    while True:
        pop.step_epoch(epoch, mutation_rate, mutation_p, mean_offs = mean_offs, mean_parity = mean_parity, num_timesteps = num_timesteps, **kwargs)
        top_inds = pop.show_top(8)        
        res = input("keep genome?")
        if res:
            
            try:
                ni = int(res)
            except:
                continue
            
            indis.append(top_inds[ni])
            
            if len(indis) > 1:
                return indis


def main():
    
    ploidy = 2
    num_genes = 9
    density = 0.7
    resource = num_genes/2
    
    num_phenotypes = 2
    pheno_density = 1.0
    
    mean_scale = 1.5
    sd_scale = 0.5
    mean_threshold = 0.2
    sd_threshold = 0.4
    mean_decay = 0.5
    sd_decay = 0.2
    mean_weight = 0.0
    sd_weight = 1.5
    
    mean_pheno_weight = 0.2
    sd_pheno_weight = 0.2
    
    mean_init_product = 0.5
    sd_init_product = 0.25
    
    generator = "normal"
    
    genome_kwargs = dict(ploidy = ploidy,
        resource = resource,
        
        generator = generator,
        mean_scale = mean_scale,
        sd_scale = sd_scale,
        mean_threshold = mean_threshold,
        sd_threshold = sd_threshold,
        mean_decay = mean_decay,
        sd_decay = sd_decay,
        mean_weight = mean_weight,
        sd_weight = sd_weight,
        
        pheno_density = pheno_density,
        mean_pheno_weight = mean_pheno_weight,
        sd_pheno_weight = sd_pheno_weight,
        
        mean_init_product = mean_init_product,
        sd_init_product = sd_init_product
    )
    
    num_bridges = 2
    bridge_density = 0.5
    
    mean_br_sc = 1.0
    sd_br_sc = 0.2
    mean_br_thr = 0.3
    sd_br_thr = 0.2
    mean_br_decay = 0.4
    sd_br_decay = 0.2
    mean_br_wgt = 0.3
    sd_br_wgt = 0.1
    
    bridge_kwargs = dict(mean_weight = mean_br_wgt,
        sd_weight = sd_br_wgt,
        mean_scale = mean_br_sc,
        sd_scale = sd_br_sc,
        mean_threshold = mean_br_thr,
        sd_threshold = sd_br_thr,
        mean_decay = mean_br_decay,
        sd_decay = sd_br_decay,
    )
    
    num_indis = 8
    t_max = 20
    
    gnm1_kwargs = genome_kwargs.copy()
    org1 = try_genomes(num_indis, t_max, num_genes, num_phenotypes, density, **gnm1_kwargs)
    gnm1 = org1.genome
    
    gnm2_kwargs = genome_kwargs.copy()
    
    org2 = try_genomes(num_indis, t_max, num_genes, num_phenotypes, density, **gnm2_kwargs)
    gnm2 = org2.genome
    
    print("selected genomes:")
    org1.show_genome()
    org2.show_genome()
    
    mgnm = make_merged_genome(gnm1, gnm2, num_bridges, bridge_density, **bridge_kwargs)
    
    rate = 0.1
    p = 0.5
    merged_resource = resource * 2
    
    indi_kwargs = genome_kwargs.copy()
    indi_kwargs.pop("resource")
    
    indi = try_individuals(mgnm, num_indis, rate, p, merged_resource, **indi_kwargs)
    
    # indi = try_individuals(mgnm, num_indis, rate, p, merged_resource, t_max = t_max, **indi_kwargs)
    # indi = try_merged_genomes(num_indis, t_max, num_genes, num_phenotypes, density, num_bridges, bridge_density, merged_resource, **indi_kwargs)
    
    pop_size = 200
    num_timesteps = 20
    mean_offs = 2.5
    mean_parity = 4
    epoch = 5
    
    new_indis = try_generation(indi.genome, pop_size, num_timesteps, rate, p, merged_resource, epoch = epoch, mean_offs = mean_offs, mean_parity = mean_parity, **indi_kwargs)
    
    gnm1 = new_indis[0].genome
    gnm2 = new_indis[1].genome
    
    print("selected genomes:")
    new_indis[0].show_genome()
    new_indis[1].show_genome()
    
    mgnm = make_merged_genome(gnm1, gnm2, num_bridges, bridge_density, **bridge_kwargs)
    
    indi = try_individuals(mgnm, num_indis, rate, p, merged_resource, **indi_kwargs)
    
    new_indis = try_generation(indi.genome, pop_size, num_timesteps, rate, p, merged_resource, epoch = epoch, mean_offs = mean_offs, mean_parity = mean_parity, **indi_kwargs)
    # num_genomes = 4
    # t_max = 24
    # try_genomes(num_genomes, num_genes, num_phenotypes, density, **genome_kwargs)


if __name__=="__main__":
    main()

