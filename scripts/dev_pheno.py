
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
    
    for n in range(num_genomes):
        
        gnm = get_genome(num_genes, num_phenotypes, density, **kwargs)
        org = Organism(gnm, resource = rsrc)
        org.step_to(t_max)
        
        orgs.append(org)
    
    orgs[0].show_genome()
    # Organism.plot_interesting_genes(*orgs, topk = 5, headers = [f"Genome{i}" for i in range(num_genomes)])
    # print("^^^^^^^^^^^^^ interesting ^^^^^^^^^^^^^^^^^^^^^^^")
    Organism.plot_expressions(*orgs, genes = True, phenos = False, headers = [f"Genome{i}" for i in range(num_genomes)])
    
    return orgs

def generate_genomes(num_genomes, t_max, num_genes, num_phenotypes, density, **kwargs):
        
    while True:
        orgs = try_genomes(num_genomes, t_max, num_genes, num_phenotypes, density, **kwargs)
        res = input("save a genome?\n")
        
        if res:
            try:
                ind = int(res)
            except:
                continue
            
            gnm = orgs[ind].genome
            state = gnm.get_state()
            
            ts = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
            
            fpath = os.path.join("./data", f"{ts}_genome.json")
            
            with open(fpath, "w+") as f:
                json.dump(state, f, indent = 3)
    

def main():
    
    ploidy = 2
    num_genes = 15
    density = 1.0
    bridge_density = 0.3
    resource = num_genes * 0.5
    
    num_phenotypes = 0
    pheno_density = 0.4
    
    num_modules = 2
    num_bridges = 1
    sd_mod_frac = 0.0
    
    mean_scale = 1.0
    sd_scale = 0.2
    mean_threshold = 0.4
    sd_threshold = 0.2
    mean_decay = 0.4
    sd_decay = 0.2
    mean_weight = 0.3
    sd_weight = 2.0
    
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
        
        num_modules = num_modules,
        num_bridges = num_bridges,
        bridge_density = bridge_density,
        sd_mod_frac = sd_mod_frac,
        
        mean_init_product = mean_init_product,
        sd_init_product = sd_init_product
    )
    
    num_genomes = 4
    t_max = 48
    
    # try_genomes(num_genomes, num_genes, num_phenotypes, density, **genome_kwargs)
    generate_genomes(num_genomes, t_max, num_genes, num_phenotypes, density, **genome_kwargs)


if __name__=="__main__":
    main()


