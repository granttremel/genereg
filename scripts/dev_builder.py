
import numpy as np

from tabulate import tabulate

from scipy.stats import describe

from genereg.network import gene, genome
from genereg.network.gene import Gene, Allele, Phene, Phallele, Interaction

from genereg.network.genome import Genome

from genereg import utils, organism
from genereg.organism import Organism, Population
from genereg.build.builder import GenomeBuilder

from genereg.build.params import GeneParams, from_custom_params

def test_params():
    
    gps = GeneParams.load_defaults()
    pps = GeneParams.load_params(fname = "phene_params")
    bps = GeneParams.load_params(fname = "bridge_params")
    print(gps)
    print(pps)
    print(bps)

def test_builder():
    num_genomes = 8
    num_timesteps = 24
    
    num_branches = 1
    num_layers = 1
    
    b = GenomeBuilder()
    # b = GenomeBuilder.from_custom_params("cust1")
    
    # b.genome_params.num_genes = 5
    b.pop_params.num_epochs = 0
    gnms = b.generate_select(num_genomes, num_timesteps, num_branches, num_layers, genes = True, shuffle_mode = "genome")
    
    gnms[0].print()
    gnms[0].show_interactions()
    gnms[0].show_interaction_heatmap()
    # input()
    return
    
    b.pop_params.num_epochs = 10
    topk = b._evolve(gnms[0])

    pgen =b.gene_params.get_generator()
    init_product = lambda : pgen(b.gene_params.mean_init_product, b.gene_params.sd_init_product)
    for gnm in topk:
        ind = Organism(gnm, resource = b.genome_params.resource)
        ind.initialize(init_product = init_product)
        ind.step_to(20)
        ind.genome.show_interaction_heatmap()
        ind.plot_expression(genes = True)
        input()
    

def main():


    # test_colors()
    
    # test_params()
    
    test_builder()
    
    
    
    pass

    

if __name__=="__main__":
    main()


