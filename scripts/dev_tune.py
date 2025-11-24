

import numpy as np

from tabulate import tabulate

from scipy.stats import describe

from genereg.network import gene, genome
from genereg.network.gene import Gene, Allele, PhenotypicGene, PhenotypicAllele

from genereg.network.genome import Genome

from genereg import utils, organism
from genereg.organism import Organism, Population
from genereg.tune import Tuner

def get_genome(num_genes, num_phenotypes, num_interactions, **kwargs):
    
    gnm = Genome.initialize_random(num_genes, num_phenotypes, num_interactions, **kwargs)
    return gnm

def main():
    
    # gnm = test_random(10, 50)
    # org = test_organism(gnm)
    
    pop_size = 100
    
    num_rounds = 100
    num_genes = 10
    num_inters = 81
    num_phenotypes = 0
    
    num_modules = 2
    sd_mod_frac = 0.1
    
    num_timesteps = 20
    
    mean_expression = 0.1
    sd_expression = 0
    mean_threshold = 0.5
    sd_threshold = 0.0
    mean_decay = 0.3
    sd_decay = 0.00
    mean_weight = 0.5
    sd_weight = 0.0
    
    mean_init_product = 0.1
    sd_init_product = 0.0
    
    gnm = get_genome(num_genes, num_phenotypes, num_inters,
                            mean_expression = mean_expression,
                            sd_expression = sd_expression,
                            mean_threshold = mean_threshold,
                            sd_threshold = sd_threshold,
                            mean_decay = mean_decay,
                            sd_decay = sd_decay,
                            mean_weight = mean_weight,
                            sd_weight = sd_weight,
                            
                            num_modules = num_modules,
                            sd_mod_frac = sd_mod_frac,
                            
                            mean_init_product = mean_init_product,
                            sd_init_product = sd_init_product,
                        )


    tuner = Tuner(gnm)
    
    t_meas = 20
    rate = 0.02
    
    sens = tuner.calculate_sensitivity(t_meas, rate=rate)
    
    print(sens)
    # lbls = [["Allele"]]
    
    for i in range(len(sens)):
        print(i, format(sens[i], '0.3f'))


if __name__=="__main__":
    main()


