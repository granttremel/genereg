
import numpy as np

from tabulate import tabulate

from scipy.stats import describe

from genereg.network import gene, genome
from genereg.network.gene import Gene, Allele, PhenotypicGene, PhenotypicAllele

from genereg.network.genome import Genome

from genereg import utils, organism
from genereg.organism import Organism, Population


def test_random(num_genes, num_interactions, num_phenotypes = 3):
    
    gnm = Genome.initialize_random(num_genes, num_phenotypes, num_interactions,
                                   mean_expression = 0,
                                   sd_expression = 0,
                                   mean_threshold = 0.2,
                                   sd_threshold = 0.5,
                                   mean_decay = 0.05,
                                   sd_decay = 0.01,
                                   mean_weight = 0.0,
                                   sd_weight = 1.0,
                                )
    
    print(repr(gnm))
    
    inters = gnm.get_interactions()
    state = gnm.get_state()
    
    return gnm

def test_organism(genome):
    
    org = Organism(genome)
    
    org.initialize(
        mean_init_product = 0.1, 
        sd_init_product = 0.0,
        
    )
    org.show_genome()
    org.step_to(30)
    org.show_genome()
    
    # org.show_history(show_product = False)
    
    org.plot_expression()
    # org.plot_product()
    
    return org

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

def try_generation(pop_size, num_genes, num_inters, num_phenotypes, num_timesteps, **kwargs):
    
    
    mean_init_product = kwargs.get("mean_init_product", 0.1)
    sd_init_product = kwargs.get("sd_init_product", 0.0)
    
    gnm = Genome.initialize_random(num_genes, num_phenotypes, num_inters, **kwargs)
    pop = Population(gnm)
    
    for n in range(pop_size):
        org = Organism(gnm.randomize_alleles(**kwargs))
        pop.add_individual(org)
    
    gen = 0
    while True:
        pop.individuals[0].show_genome()
        pop.initialize(mean_init_product=mean_init_product, sd_init_product=sd_init_product)
        pop.step_to(num_timesteps)
        
        mean_offs = kwargs.get("mean_offs", 4)
        n_parents = pop_size // mean_offs
        
        topk_orgs, topk_uniqs = pop.quantify(topk = n_parents)
        
        topk_orgs[0].show_genome()
        topk_orgs[0].plot_expression()
        
        new_offs = pop.step_generation(topk_orgs, mean_offs, mutation_rate = 0.3, mutation_p = 0.3)
        gen += 1
        
        res = input("keep going?\n")
        if 'n' in res.lower():
            break
    
    print("most unique organism (gen 1)")
    topk_orgs[0].show_genome()
    topk_orgs[0].plot_expression()


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
        
        # res = input("keep going?")
        # if 'n' in res.lower():
        #     break
    
    # try_generation(pop_size, num_genes, num_inters, num_phenotypes, num_timesteps, 
    #                 mean_expression = mean_expression,
    #                 sd_expression = sd_expression,
    #                 mean_threshold = mean_threshold,
    #                 sd_threshold = sd_threshold,
    #                 mean_decay = mean_decay,
    #                 sd_decay = sd_decay,
    #                 mean_weight = mean_weight,
    #                 sd_weight = sd_weight,
                    
    #                 mean_init_product = mean_init_product,
    #                 sd_init_product = sd_init_product,
    #             )
    
    # print(org.history["expression"])
    # print(org.history["product"])
    


if __name__=="__main__":
    main()


