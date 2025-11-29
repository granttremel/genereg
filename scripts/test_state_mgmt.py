
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

from genereg.tune import Tuner

def test_state():
    
    gnm = Genome.initialize_random(10, 0, 0.6)
    org = Organism(gnm)
    org.initialize()
    org.step_to(20)
    
    org.show_genome()
    org.plot_expression()
    org.plot_states()
    
    gnm_state = gnm.get_state()
    
    # print(json.dumps(gnm_state, indent = 3))
    
    new_gnm = gnm.randomize_alleles()
    new_org = Organism(new_gnm)
    new_org.initialize()
    new_org.step_to(20)
    
    print("before setting state")
    new_org.show_genome()
    org.plot_expression()
    
    new_org.genome.set_state(gnm_state)
    
    print("after setting state")
    new_org.show_genome()
    
    return gnm

def test_tuner():
    
    gnm = Genome.initialize_random(3, 0, 0.6)
    org = Organism(gnm)
    org.initialize()
    
    tuner = Tuner(gnm)
    
    org.step_to(10)
    tuner.save_state(state_key = "t10")
    
    sd = tuner.state_vector_to_dict(tuner.init_state)
    print(sd)

def test_sens():
    
    gnm = Genome.initialize_random(3, 0, 0.6)
    gnm.initialize()
    
    tuner = Tuner(gnm)
    
    state_sigs = tuner.measure_noise_sensitivity(0.02, num_tests = 10, t_max = 20)
    
    print(json.dumps(state_sigs, indent = 3))

def get_genome(num_genes, num_phenotypes, density, **kwargs):
    
    mean_init_product = kwargs.pop("mean_init_product", 0.1)
    sd_init_product = kwargs.pop("sd_init_product", 0.0)
    
    gnm = Genome.initialize_random(num_genes, num_phenotypes, density, **kwargs)
    
    gnm.initialize(mean_init_product = mean_init_product, sd_init_product = sd_init_product)
    
    return gnm

def try_genomes(num_genomes, num_genes, num_phenotypes, density, **kwargs):
    
    orgs = []
    
    rsrc = kwargs.get("resource", -1)
    
    for n in range(num_genomes):
        
        gnm = get_genome(num_genes, num_phenotypes, density, **kwargs)
        org = Organism(gnm, resource = rsrc)
        org.step_to(20)
        
        orgs.append(org)
    
    orgs[0].show_genome()
    Organism.plot_expressions(*orgs, headers = [f"Genome{i}" for i in range(num_genomes)])
    
    return orgs
    

def main():
    
    # gnm = test_state()
    
    # test_tuner()
    
    ploidy = 2
    num_genes = 15
    density = 1.0
    resource = num_genes / 3
    num_phenotypes = 0
    
    num_modules = 2
    num_bridges = 1
    sd_mod_frac = 0.2
    
    mean_scale = 1.0
    sd_scale = 0.3
    mean_threshold = 0.3
    sd_threshold = 0.3
    mean_decay = 0.3
    sd_decay = 0.1
    mean_weight = 0.05
    sd_weight = 2
    
    mean_init_product = 0.5
    sd_init_product = 0.3
    
    generator = "normal"
    
    while True:
        orgs = try_genomes(8, num_genes, num_phenotypes, density, 
                            ploidy = ploidy,
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
                            
                            num_modules = num_modules,
                            num_bridges = num_bridges,
                            sd_mod_frac = sd_mod_frac,
                            
                            mean_init_product = mean_init_product,
                            sd_init_product = sd_init_product
                        )
        res = input("save a genome?\n")
        
        if res:
            ind = int(res)
            
            gnm = orgs[ind].genome
            state = gnm.get_state()
            
            ts = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
            
            fpath = os.path.join("./data", f"{ts}_genome.json")
            
            with open(fpath, "w+") as f:
                json.dump(state, f, indent = 3)
            
            
    
    # test_sens()
    
    
    
    
    pass



if __name__=="__main__":
    main()


