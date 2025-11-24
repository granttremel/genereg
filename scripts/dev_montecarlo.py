
import random
import numpy as np

from tabulate import tabulate

from scipy.stats import describe

from genereg.network import gene, genome
from genereg.network.gene import Gene, Allele, PhenotypicGene, PhenotypicAllele

from genereg.network.genome import Genome

from genereg import utils, organism
from genereg.organism import Organism, Population

def try_round(num_orgs, num_genes, num_interactions, num_phenotypes, num_timesteps, **kwargs):
    mean_init_product = kwargs.pop("mean_init_product", 0.1)
    sd_init_product = kwargs.pop("sd_init_product", 0.0)
    
    gnm = Genome.initialize_random(num_genes, num_phenotypes, num_interactions, **kwargs)
    orgs = []
    for no in range(num_orgs):
        
        rand_gnm = gnm.randomize_alleles(**kwargs)
        org = Organism(rand_gnm)
        org.initialize(
            mean_init_product = mean_init_product, 
            sd_init_product = sd_init_product,
        )
        org.step_to(num_timesteps)
        inch, uniq = org.quantify()
        orgs.append(org)
    
    
    orgs[0].show_genome()
    print(tabulate(orgs[0].genome.get_interactions()))
    orgs[0].plot_product()
    Organism.plot_expressions(*orgs)
    
    # print(orgs[0].history.get("product"))
    
    pass

def try_monte_carlo(num_rounds, num_genes, num_interactions, num_phenotypes, num_timesteps, **kwargs):
    
    mean_init_product = kwargs.pop("mean_init_product", 0.1)
    sd_init_product = kwargs.pop("sd_init_product", 0.0)
    
    topk = kwargs.get("topk", 5)
    
    data = []
    
    # ind_max_inch = -1
    # max_inch = 0
    # most_inch = None
    # ind_max_uniq = -1
    # max_uniq =0
    # most_uniq = None
    
    for nr in range(num_rounds):
        gnm = Genome.initialize_random(num_genes, num_phenotypes, num_interactions, **kwargs)
        
        org = Organism(gnm)
        org.initialize(
            mean_init_product = mean_init_product, 
            sd_init_product = sd_init_product,
        )
        
        org.step_to(num_timesteps)
        
        inch, uniq = org.quantify()
        data.append((org, inch, uniq))
        
        # if inch > max_inch:
        #     most_inch = org
        #     max_inch = inch
        #     ind_max_inch = nr
        # if uniq > max_uniq:
        #     most_uniq = org
        #     max_uniq = uniq
        #     ind_max_uniq = nr
    
    inch_data = sorted(data, key = lambda k: -k[1])
    uniq_data = sorted(data, key = lambda k: -k[2])
    
    inch_res = describe([inch for org, inch, uniq in inch_data])
    uniq_res = describe([uniq for org, inch, uniq in uniq_data])
    
    rows = []
    rows.append(["Interestingness",inch_res.mean, np.sqrt(inch_res.variance), inch_res.minmax[0], inch_res.minmax[1]])
    rows.append(["Uniqueness",uniq_res.mean, np.sqrt(uniq_res.variance), uniq_res.minmax[0], uniq_res.minmax[1]])
    
    print(tabulate(rows, headers = ["", "Mean", "SD","Min","Max"], floatfmt = "0.3f"))
    print()

    # print([org for org, _, _ in inch_data[:topk]])

    print("Most interesting:")
    Organism.plot_expressions(*[org for org, _, _ in inch_data[:topk]])
        
    # res = input("save genome?")
    # if 'y' in res.lower():
    #     fname = f"genome_{ind_max_inch}_interesting.json"
    #     most_inch.genome.save_state(fname, interestingness = max_inch)
    
    print("Most unique:")
    Organism.plot_expressions(*[org for org, _, _ in uniq_data[:topk]])
    
    # res = input("save genome?")
    # if 'y' in res.lower():
    #     fname = f"genome_{ind_max_uniq}_unique.json"
    #     most_uniq.genome.save_state(fname, uniqueness = max_uniq)
        
    # print(org.genome.get_interactions())
    # tab = [inters for inters in org.genome.get_interactions()]
    # print(tabulate(tab))
    
    return gnm

def try_basic():
    
    gnm = Genome()
    scale = 0.5
    thr = 0.1
    mean_decay = 0.3
    sd_decay = 0.05
    
    g1 = Gene("Gene1", epistasis_type = "max")
    a11 = g1.create_allele("Allele1-1", scale, thr, random.normalvariate(mean_decay, sd_decay))
    a12 = g1.create_allele("Allele1-2", scale, thr, random.normalvariate(mean_decay, sd_decay))
    
    g2 = Gene("Gene2", epistasis_type = "max")
    a21 = g2.create_allele("Allele2-1", scale, thr, random.normalvariate(mean_decay, sd_decay))
    a22 = g2.create_allele("Allele2-2", scale, thr, random.normalvariate(mean_decay, sd_decay))
    
    g3 = Gene("Gene3", epistasis_type = "max")
    a31 = g3.create_allele("Allele3-1", scale, thr, random.normalvariate(mean_decay, sd_decay))
    a32 = g3.create_allele("Allele3-2", scale, thr, random.normalvariate(mean_decay, sd_decay))
    
    mean_exp = 0.5
    sd_exp = 0.1
    
    gnm.add_gene(g1, a11, a12, expression = random.normalvariate(mean_exp, sd_exp))
    gnm.add_gene(g2, a21, a22, expression = random.normalvariate(mean_exp, sd_exp))
    gnm.add_gene(g3, a31, a32, expression = random.normalvariate(mean_exp, sd_exp))
    
    mean_wgt = 0.5
    sd_wgt = 0.01
    gnm.add_interaction(g1, g2, random.normalvariate(mean_wgt, sd_wgt))
    gnm.add_interaction(g1, g3, -random.normalvariate(mean_wgt, sd_wgt))
    
    gnm.add_interaction(g2, g3, -random.normalvariate(mean_wgt, sd_wgt))
    gnm.add_interaction(g2, g1, random.normalvariate(mean_wgt, sd_wgt))
    
    gnm.add_interaction(g3, g1, random.normalvariate(mean_wgt, sd_wgt))
    gnm.add_interaction(g3, g2, -random.normalvariate(mean_wgt, sd_wgt))
    
    org = Organism(gnm)
    
    mean_prod = 0.2
    sd_prod = 0.05
    prods = [random.normalvariate(mean_prod, sd_prod) for i in range(6)]
    prods[0] = 5
    
    org.initialize(init_product = prods)
    org.show_genome()
    print(tabulate(org.genome.get_interactions()))
    
    org.step_to(20)
    org.show_genome()
    org.plot_product()
    org.plot_expression()
    
    


def main():
    
    # gnm = test_random(10, 50)
    # org = test_organism(gnm)
    
    num_orgs = 5
    pop_size = 100
    
    num_rounds = 100
    num_genes = 10
    num_inters = (num_genes) * (num_genes - 1) // 2
    num_phenotypes = 0
    
    num_modules = 2
    sd_mod_frac = 0.1
    
    num_timesteps = 20
    
    mean_expression = 0.5
    sd_expression = 0.1
    
    mean_scale = 0.5
    sd_scale = 0.1
    
    mean_threshold = 0.3
    sd_threshold = 0.1
    
    mean_decay = 0.05
    sd_decay = 0.01
    
    mean_weight = 0.5
    sd_weight = 3.0
    
    mean_init_product = 1.0
    sd_init_product = 0.2
    
    while True:
        try_monte_carlo(num_rounds, num_genes, num_inters, num_phenotypes, num_timesteps,
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
        input()
    
    # try_round(num_orgs, num_genes, num_inters, num_phenotypes, num_timesteps,
    #                     mean_expression = mean_expression,
    #                     sd_expression = sd_expression,
    #                     mean_scale = mean_scale,
    #                     sd_scale = sd_scale,
    #                     mean_threshold = mean_threshold,
    #                     sd_threshold = sd_threshold,
    #                     mean_decay = mean_decay,
    #                     sd_decay = sd_decay,
    #                     mean_weight = mean_weight,
    #                     sd_weight = sd_weight,
                        
    #                     num_modules = num_modules,
    #                     sd_mod_frac = sd_mod_frac,
                        
    #                     mean_init_product = mean_init_product,
    #                     sd_init_product = sd_init_product,
    #                 )

    # try_basic()



if __name__=="__main__":
    main()


