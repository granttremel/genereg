
from typing import Dict, List, Tuple, Any, Union, Optional
import numpy as np
import random

from genereg.utils import Rectifier, CostFunction, Epistasis

class Gene:
    """
    a gene that represents a node in a gene regulatory network. the gene product may influence other genes, produce functional products (i.e. parameters), and may respond to environmental conditions. the gene represents the functional role and presence in genome of a gene, and an instance of a gene is represented by an allele. the gene represents in some sense the rules that each allele has to play by while competing or cooperating to become enriched in the population. 
    a genome contains n copies of gene g, where n is the ploidy, but may contain several copies of the gene which have strictly associated alleles, and may diverge if they adapt to fill different functional niches of the organism
    """
    
    def __init__(self, name, cost = "parabolic", cost_factor = 1.0, rect = "Sigmoid", epistasis_type = "sum", **params):

        self.name = name
        self.alleles:Dict[str, 'Allele'] = {}
        
        self._params = params
        self.rect:Rectifier = Rectifier(rect, **params)
        
        self.cost_factor = np.clip(cost_factor, 0.0, None) # like weight in zemax
        self.cost_func = CostFunction(cost_func = cost)
        
        self.epistasis = Epistasis(epistasis_type)
    
    def rectify(self, gene_product):
        return self.rect(gene_product)
    
    def evaluate(self, *allele_products):
        gene_product = self.epistasis(*allele_products)
        return self.rectify(gene_product)
    
    def create_allele(self, allele_id, scale, threshold, decay):
        allele_id = f"Allele{len(self.alleles)}"
        a = Allele(self, allele_id, scale=scale, threshold=threshold, decay=decay)
        self.add_allele(a)
        return a

    def add_allele(self, allele):
        self.alleles[allele.id] = allele


class Allele:
    """
    an allele, which represents an instance of a gene. the allele represents a unique approach to filling the functional role of the gene, and so interacts individually with the other products of the genome and the organism. 
    """
    
    def __init__(self, gene:Gene, allele_id, scale, threshold, decay):
        
        self.gene = gene
        self.id = allele_id
        self.scale = np.clip(scale, 0.0, None)
        self.threshold = np.clip(threshold, 0.0, None)
        self.decay = np.clip(decay, 0.0, 1.0)
        self.product = 0.0
        
        self.silenced = False
    
    def initialize(self, init_product = 0.0):
        self.product = init_product
    
    def evaluate(self, expression):
        """
        conversion of gene expression quantity into measure of gene product. expression is added to baseline expression, i.e. mRNAs per time, then scaled to quantity of gene product (e.g. concentration of protein). Gene products degrade at a rate determined by decay, and residual is added. This is rectified in gene-dependent manner to maintain biological realism (some gene products cannot be negative, some have saturating behavior). 
        """
        gene_product = self.scale * (expression-self.threshold) + self.decay * self.product
        self.product = gene_product
        # return self.gene.rectify(gene_product)
        return gene_product
        
    def get_params(self):
        return np.array([self.scale, self.threshold, self.decay])
    
    def set_params(self, param_vec):
        
        self.scale = param_vec[0]
        self.threshold = param_vec[1]
        self.decay = param_vec[2]
        
        return self.get_params()
    
    def toggle_silence(self, is_silent = True):
        self.silenced = is_silent
    
    def copy(self):
        return Allele(self.gene, self.id, self.scale, self.threshold, self.decay)
    
    def mutate(self, rate = 0.1, p = 0.2):
        mscale = self.scale
        mthreshold = self.threshold
        mdecay = self.decay

        if random.random() < p:
            # Use a small epsilon to avoid zero variance when scale is 0
            mscale += random.normalvariate(0, rate * max(abs(self.scale), 0.01))
        if random.random() < p:
            # Allow threshold to mutate even when 0
            mthreshold += random.normalvariate(0, rate * max(abs(self.threshold), 0.01))
        if random.random() < p:
            # Ensure decay stays between 0 and 1
            mdecay += random.normalvariate(0, rate * max(self.decay, 0.01))
            mdecay = max(0, min(1, mdecay))  # Clamp between 0 and 1

        return Allele(self.gene, self.id, mscale, mthreshold, mdecay)
    
class PhenotypicGene(Gene):
    """
    this is a gene whose product has a direct effect on phenotype, mediated by no regulatory mechanisms. the cost is defined by thermodynamics or environment and is not subject to balancing. if the phenotype is fundamentally produced by interactions of genes (almost always true), then this gene is symbolic, not an element of the genome but representative of the process of integrating genes into phenotype
    
    """
    
    def __init__(self, *args, min_value, max_value, epistasis_type = "mean", **kwargs):
        super().__init__(*args, **kwargs)
        self.rect = Rectifier("Sigmoid")
        self.min_value = min_value
        self.max_value = max_value
        
        self.epistasis = Epistasis(epistasis_type)
    
    def rectify(self, gene_product):
        return self.rect.apply_rescale(gene_product, self.min_value, self.max_value)
    
    def evaluate(self, *allele_products):
        gene_product = self.epistasis(*allele_products)
        return self.rectify(gene_product)
    
    def create_allele(self, allele_id, scale, baseline, decay):
        a = PhenotypicAllele(self, allele_id, scale, baseline, decay)
        self.add_allele(a)
        return a
    
class PhenotypicAllele(Allele):
    
    def __init__(self, pgene:PhenotypicGene, *args):
        if not isinstance(pgene, PhenotypicGene):
            raise ValueError
        super().__init__(pgene, *args)
    
    def evaluate(self, expression, current):
        return self.scale*(expression+self.threshold) + (1 - self.decay) * current