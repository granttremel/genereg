
import numpy as np

from tabulate import tabulate

from scipy.stats import describe

from genereg.network import gene, genome
from genereg.network.gene import Gene, Allele, PhenotypicGene, PhenotypicAllele

from genereg.network.genome import Genome

from genereg import utils, organism
from genereg.organism import Organism, Population


def try_generation(pop_size, num_genes, num_phenotypes, density, num_timesteps, **kwargs):
    
    
    mean_init_product = kwargs.get("mean_init_product", 0.1)
    sd_init_product = kwargs.get("sd_init_product", 0.0)
    
    epoch = kwargs.get("epoch", 20)
    
    gnm = Genome.initialize_random(num_genes, num_phenotypes, density, **kwargs)
    pop = Population(gnm)
    
    for n in range(pop_size):
        org = Organism(gnm.randomize_alleles(**kwargs))
        pop.add_individual(org)
    
    pop.individuals[0].show_genome()
    
    # input()
    
    gen = 0
    while True:
        pop.initialize(mean_init_product=mean_init_product, sd_init_product=sd_init_product)
        pop.step_to(num_timesteps)
        
        mean_offs = kwargs.get("mean_offs", 4)
        n_parents = pop_size // mean_offs
        
        topk_orgs, topk_uniqs = pop.quantify(topk = n_parents)
        
        new_offs = pop.step_generation(topk_orgs, mean_offs, mutation_rate = 0.3, mutation_p = 0.3)
        gen += 1
        
        print(f"completed gen {gen}, with pop size {len(pop.individuals)} mean uniqueness {np.mean(topk_uniqs):0.3f}")
        
        if gen % epoch == 0:
            topk_orgs[0].show_genome()
            Organism.plot_expressions(*topk_orgs[:8])
            res = input("keep going?\n")
            if 'n' in res.lower():
                break
    
    print("most unique organism (gen 1)")
    topk_orgs[0].show_genome()
    # topk_orgs[0].plot_expression()
    Organism.plot_expressions(*topk_orgs)


def main():
    
    
    num_orgs = 5
    pop_size = 100
    
    num_rounds = 100
    epoch = 10
    
    num_genes = 10
    density = 0.8
    num_phenotypes = 0
    
    num_modules = 1
    num_bridges = 1
    sd_mod_frac = 0.1
    
    num_timesteps = 20
    
    mean_expression = 0.5
    sd_expression = 0.1
    
    mean_scale = 0.8
    sd_scale = 0.02
    
    mean_threshold = 0.0
    sd_threshold = 0.1
    
    mean_decay = 0.1
    sd_decay = 0.03
    
    mean_weight = 0.2
    sd_weight = 1.5
    
    mean_init_product = 0.5
    sd_init_product = 0.05
    
    try_generation(pop_size, num_genes, num_phenotypes, density, num_timesteps,
                   epoch = epoch, 
                    mean_expression = mean_expression,
                    sd_expression = sd_expression,
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
                    sd_init_product = sd_init_product,
                )
    # org = test_organism(gnm)


if __name__=="__main__":
    main()


