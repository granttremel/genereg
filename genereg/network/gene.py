
from typing import Dict, List, Tuple, Any, Union, Optional
import numpy as np
import random

from genereg.utils import Rectifier, CostFunction, Epistasis

class Gene:
    """
    a gene that represents a node in a gene regulatory network. the gene product may influence other genes, produce functional products (i.e. parameters), and may respond to environmental conditions. the gene represents the functional role and presence in genome of a gene, and an instance of a gene is represented by an allele. the gene represents in some sense the rules that each allele has to play by while competing or cooperating to become enriched in the population. 
    a genome contains n copies of gene g, where n is the ploidy, but may contain several copies of the gene which have strictly associated alleles, and may diverge if they adapt to fill different functional niches of the organism
    """
    
    _name_frm = "{name}-{tag}"
    
    def __init__(self, name = "", cost = "parabolic", cost_factor = 1.0, rect = "Sigmoid", epistasis_type = "sum", **params):

        self._base_name = name
        self.tag = params.get("tag","")
        
        self.alleles:Dict[str, 'Allele'] = {}
        
        self.gene_product = 0
        self.regulation = 0
        
        self._params = params
        self.rect:Rectifier = Rectifier(rect, **params)
        
        self.cost_factor = np.clip(cost_factor, 0.0, None) # like weight in zemax
        self.cost_func = CostFunction(cost_func = cost)
        
        self.epistasis = Epistasis(epistasis_type)
        self.old_name = ""
        
        self.is_bridge = params.get("is_bridge", False)
    
    @property
    def name(self):
        if self.tag:
            return self._name_frm.format(name=self._base_name, tag=self.tag)
        else:
            return self._base_name 
        
    @property
    def is_phenotype(self):
        return type(self).__name__ == "Phene"
    
    @property
    def is_genotype(self):
        return not self.is_phenotype
    
    def rectify(self, gene_product):
        return self.rect(gene_product)
    
    def evaluate(self, *allele_products):
        if None in allele_products:
            print(self.name, [a.id for a in self.alleles.values()])
            print([a.get_state() for a in self.alleles.values()])
            return 0
        gene_product = self.epistasis(*allele_products)
        self.gene_product = self.rectify(gene_product)
        return self.gene_product
    
    def calculate_cost(self, allele_product):
        return self.cost_func(allele_product, self.cost_factor)
    
    def scale_expression(self, scale):
        self.gene_product *= scale
    
    def copy(self):
        return self.from_state(self.get_state())
    
    def rename(self, new_name = "", tag = ""):
        old_name = self._base_name
        if new_name:
            self._base_name = new_name
        if tag:
            self.tag = tag
        self.old_name = self._name_frm.format(name=old_name, tag = self.tag)
    
    def get_state(self):
        
        return {
            "name":self._base_name,
            "tag":self.tag,
            "gene_type":type(self).__name__,
            "rect_type":self.rect.func_type,
            "rect_params":self._params,
            "cost_type":self.cost_func.func_type,
            "cost_factor":self.cost_factor,
            "epistasis_type":self.epistasis.func_type,
            "gene_product":self.gene_product,
            "wijaj":self.regulation,
        }
    
    def set_state(self, state_dict):
        base_name = state_dict.pop("name","")
        if base_name:
            self._base_name = base_name
        
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
    
    def create_allele(self, allele_id, **kwargs):
        allele_id = f"Allele{len(self.alleles)}"
        a = Allele(self, allele_id, **kwargs)
        self.add_allele(a)
        return a

    def add_allele(self, allele:'Allele'):
        self.alleles[allele.id] = allele

    @classmethod
    def from_state(cls, state_dict, allele_dicts = []):
        
        gene = cls(state_dict.get("name"))
        gene.set_state(state_dict)
        for ad in allele_dicts:
            allele = Allele.from_state(ad, gene)
            gene.add_allele(allele)
        
        return gene


class Allele:
    """
    an allele, which represents an instance of a gene. the allele represents a unique approach to filling the functional role of the gene, and so interacts individually with the other products of the genome and the organism. 
    """
    
    _id_frm = "{id}-{tag}"
    
    def __init__(self, gene:Gene, allele_id, scale = 1.0, threshold = 0.1, decay = 0.05, **kwargs):
        
        self.gene = gene
        self._id = allele_id
        self.scale = np.clip(scale, 0.0, None)
        self.threshold = threshold
        self.decay = np.clip(decay, 0.0, 1.0)
        self.n = kwargs.get("n", 2.0)
        self.kn = kwargs.get("kn", 0.5)
        self.product = kwargs.get("init_product")
        
        self.silenced = False
    
    @property
    def id(self):
        if self.gene.tag:
            return self._id_frm.format(id = self._id, tag = self.gene.tag)
        else:
            return self._id
    @property
    def is_phenotype(self):
        return self.gene.is_phenotype
    
    @property
    def is_genotype(self):
        return self.gene.is_genotype
    
    def initialize(self, init_product = 0.0):
        self.product = init_product
    
    def evaluate(self, regulation):
        """
        conversion of gene expression quantity into measure of EFFECT of gene product. expression is added to baseline expression, i.e. mRNAs per time, then scaled to quantity of gene product (e.g. concentration of protein). Gene products degrade at a rate determined by decay, and residual is added. This is rectified in gene-dependent manner to maintain biological realism (some gene products cannot be negative, some have saturating behavior). 
        """
        rn = abs(np.pow(regulation, self.n))
        theta = rn / (self.kn + rn)
        product = theta * self.scale * (regulation-self.threshold) + self.decay * self.product
        self.product = product
        return product
    
    def calculate_cost(self):
        return self.gene.calculate_cost(self.product)
    
    def get_state(self):
        return {
            "id":self._id,
            "allele_type":type(self).__name__,
            "parent_gene":self.gene.name,
            "scale":self.scale,
            "threshold":self.threshold,
            "decay":self.decay,
            "n":self.n,
            "kn":self.kn,
            "product":self.product
        }
    
    def set_state(self, state_dict, parent_gene = None):
        
        if parent_gene:
            self.gene = parent_gene
        
        aid = state_dict.pop("id","")
        if aid:
            self._id = aid
        
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
    
    
    def toggle_silence(self, is_silent = True):
        self.silenced = is_silent
    
    def copy(self, new_gene = None):
        new_gene = new_gene or self.gene
        state_dict = self.get_state()
        return Allele.from_state(state_dict, new_gene)
    
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

        return Allele(self.gene, self.id, scale=mscale, threshold=mthreshold, decay=mdecay)

    @classmethod
    def from_state(cls, state_dict, gene):
        
        a = cls(gene, state_dict.get("id",""))
        a.set_state(state_dict)
        return a

class Interaction:
    
    def __init__(self, effector_gene, affected_gene, **kwargs):
        
        if effector_gene.is_phenotype:
            raise ValueError
        
        self.effector:Gene = effector_gene
        self.affected:Gene = affected_gene
        self.weight = kwargs.get("weight", 0.0)
        
    def evaluate(self):
        return self.weight * self.effector.gene_product
    
    def copy(self):
        return Interaction.from_state(self.get_state(), self.effector, self.affected)
    
    def get_state(self):
        return {
            "effector_name":self.effector.name,
            "affected_name":self.affected.name,
            "weight":self.weight,
        }
    
    def set_state(self, state_dict, effector_gene = None, affected_gene = None):
        
        if effector_gene:
            self.effector = effector_gene
        if affected_gene:
            self.affected = affected_gene
        
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
    
    def mutate(self, rate = 0.1, p = 0.2):
        
        weight = self.weight
        if random.random() < p:
            weight += random.normalvariate(0, rate*max(abs(self.weight), 0.01))
    
        return Interaction(self.effector, self.affected, weight = weight)
    
    @classmethod
    def from_state(cls, state_dict, effector_gene, affected_gene):
        inter = cls(effector_gene, affected_gene)
        inter.set_state(state_dict)
        return inter
    
class Phene(Gene):
    """
    this is a gene whose product has a direct effect on phenotype, mediated by no regulatory mechanisms. the cost is defined by thermodynamics or environment and is not subject to balancing. if the phenotype is fundamentally produced by interactions of genes (almost always true), then this gene is symbolic, not an element of the genome but representative of the process of integrating genes into phenotype
    
    """
    
    def __init__(self, *args, epistasis_type = "sum", **kwargs):
        super().__init__(*args, **kwargs)
        self.rect = Rectifier("ReLU")
        self.epistasis = Epistasis(epistasis_type)
    
    def rectify(self, gene_product):
        return self.rect.apply(gene_product)
    
    def evaluate(self, *allele_products):
        self.gene_product = self.epistasis(*allele_products)
        return self.gene_product
    
    def create_allele(self, allele_id, **kwargs):
        a = Phallele(self, allele_id, **kwargs)
        self.add_allele(a)
        return a
    
class Phallele(Allele):
    
    def __init__(self, pgene:Phene, *args, **kwargs):
        if not isinstance(pgene, Phene):
            raise ValueError
        super().__init__(pgene, *args, **kwargs)
    
    def evaluate(self, expression):
        return self.gene.rect(self.scale * (expression+self.threshold))