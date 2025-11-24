

from typing import Dict, List, Tuple, Any, Union, Optional
import random

import itertools

import numpy as np

import os
from pathlib import Path
import json

from genereg.network.gene import Gene, Allele, PhenotypicGene, PhenotypicAllele



class Genome:
    """
    A genome represents the set of all genes that describe an organism and the interactions between them, i.e. a gene regulatory network.
    A gene is a node, connected to other genes by edges. each edge represents a gene's influence on another, including scale and sensitivity.
    The genome may be organized into modules, 
    Two genes should interact regardless of alleles..
    """
    
    def __init__(self, ploidy = 2):
        
        self.ploidy = ploidy
        
        self.genes:List[Gene] = []
        self.expression:Dict[str,float] = {}
        
        self.genotype:List[Dict[str, Allele]]= [{} for n in range(ploidy)]
        self.phenotype:Dict[str, float]
        
        self.interactions:Dict[str, Dict[str, float]] = {}
        
        self._cache = {"expression":None, "product":None}
    
    @property
    def num_genes(self):
        return len(self.genes)
    
    @property
    def num_interactions(self):
        return len(self.interactions)
    
    def count_upstream(self):
        
        num_upstream = []
        
        for gi, gjdict in self.interactions.items():
            num_upstream.append(len(gjdict))
        
        return num_upstream
    
    def count_downstream(self):
        
        num_downstream = {gj.name:0 for gj in self.genes}
        
        for gi, gjdict in self.interactions.items():
            for gj, v in gjdict.items():
                num_downstream[gj] += 1
        
        return [num_downstream[gj.name] for gj in self.genes]
    
    def get_gene(self, gene_name, default = None):
        for g in self.genes:
            if g.name == gene_name:
                return g
        return default
    
    def add_gene(self, gene:Gene, *alleles, expression = 0):
        self.genes.append(gene)
        self.expression[gene.name] = expression
        for i in range(self.ploidy):
            self.genotype[i][gene.name] = alleles[i]
        self.interactions[gene.name] = {}
            
    def set_allele(self, allele:Allele, nchr):
        self.genotype[nchr][allele.gene.name] = allele
    
    def add_interaction(self, affected_gene:Gene, effector_gene:Gene, weight):
        if not affected_gene.name in self.interactions:
            self.interactions[affected_gene.name] = {}
        self.interactions[affected_gene.name][effector_gene.name] = weight
    
    def get_expression(self):
        return np.array([self.expression[g.name] for g in self.genes])
    
    def get_product(self):
        return np.vstack([np.array([self.genotype[chr][g.name].product for g in self.genes]) for chr in range(self.ploidy)])
    
    def get_interactions(self):
        
        inters = np.zeros((self.num_genes, self.num_genes))
        gene_names = [g.name for g in self.genes]
        
        for g1, g0dict in self.interactions.items():
            for g0, v in g0dict.items():
                g0ind = gene_names.index(g0)
                g1ind = gene_names.index(g1)
                inters[g0ind, g1ind] = v
        
        return inters
    
    def get_interaction_dict(self):
        inter_dict = {}
        gene_names = [g.name for g in self.genes]
        
        for g1, g0dict in self.interactions.items():
            for g0, v in g0dict.items():
                g0ind = gene_names.index(g0)
                g1ind = gene_names.index(g1)
                inter_dict[(g0ind, g1ind)] = v
        
        return inter_dict
    
    def get_state(self):
        
        state_vec = list()
        theta_vec = list()
        
        for gene in self.genes:
            state_vec.append(self.expression[gene.name])
            
            for n in range(self.ploidy):
                allele = self.genotype[n][gene.name]
                state_vec.append(allele.product)
                
                allele_params = allele.get_params()
                theta_vec.extend(allele_params)
            
            if gene.name in self.interactions:
                effectors = self.interactions[gene.name]
                for g in self.genes:
                    weight = effectors.get(g.name)
                    if weight is not None:
                        theta_vec.append(weight)
        
        return np.array(state_vec), np.array(theta_vec)
    
    def set_state(self, state_vec, theta_vec):
        
        state_vec = list(state_vec)
        theta_vec = list(theta_vec)
        
        for gene in self.genes:
            self.expression[gene.name] = state_vec.pop(0)
            
            for n in range(self.ploidy):
                allele = self.genotype[n][gene.name]
                allele.product = state_vec.pop(0)
                
                allele.set_params([theta_vec.pop(0) for i in range(3)])
            
            if gene.name in self.interactions:
                effectors = self.interactions[gene.name]
                for g in self.genes:
                    if g.name in effectors:
                        self.interactions[gene.name][g.name] = theta_vec.pop(0)
    
    def get_state_dict(self):
        
        state_dict = {}
        init_prod = self._cache["product"]
        
        for i,gene in enumerate(self.genes):
            state_dict[gene.name] = {"expression":self.expression[gene.name], "alleles":[]}
            for n in range(self.ploidy):
                allele = self.genotype[n][gene.name]
                state_dict[gene.name]["alleles"].append({
                    "id":allele.id,
                    "scale":allele.scale,
                    "threshold":allele.threshold,
                    "decay":allele.decay,
                    "initial_product":init_prod[n][i],
                })
        
        return state_dict
    
    def save_state(self, fname, **metadata):
        
        fpath = os.path.join("./data", fname)
        
        sd = self.get_state_dict()
        
        out_dict = {
            "state":sd,
            "meta":metadata
        }
        
        with open(fpath, "w") as f:
            json.dump(out_dict, f, indent = 3)
        
    
    def update_expression(self):

        expr = self.expression
        new_exprs = {}

        for gi in self.genes:
            inters_i = self.interactions[gi.name]

            # Calculate the weighted sum of interactions once (same for all chromosomes)
            # sum w_ij * a_j, w is interaction, a is expression
            wijaj = np.sum([inter_ij*expr[gj] for gj, inter_ij in inters_i.items()])

            # Evaluate each chromosome's allele independently
            allele_products = []
            for chr in range(self.ploidy):
                allele = self.genotype[chr][gi.name]
                allele_product = allele.evaluate(wijaj)
                allele_products.append(allele_product)

            # Combine allele products using the gene's epistasis function
            new_exprs[gi.name] = gi.evaluate(*allele_products)

        self.expression = new_exprs
        return self.get_expression()
    
    def initialize(self, init_product = None, mean_init_product = 0.2, sd_init_product = 0.02):

        if init_product is None or not init_product:
            init_product = [random.normalvariate(mean_init_product, sd_init_product) for i in range(self.num_genes * self.ploidy)]
        
        for chr in self.genotype:
            for gn, a in chr.items():
                a.initialize(init_product.pop(0))
        
        self._cache["expression"] = self.expression
        self._cache["product"] = self.get_product()
    
    def silence_gene(self, nchr, gene_name):
        allele = self.genotype[nchr][gene_name]
        
        if isinstance(allele, PhenotypicAllele):
            # at least one should be active or the organism perishes
            pass
        
        allele.toggle_silence()
    
    def unsilence_gene(self, nchr, gene_name):        
        allele = self.genotype[nchr][gene_name]
        
        if isinstance(allele, PhenotypicAllele):
            # at least one should be active or the organism perishes
            pass
        
        allele.toggle_silence(is_silent = False)
    
    def get_gamete(self, mutation_rate = 0.1, mutation_p = 0.2):
        
        gam = {}
        
        for g in self.genes:
            na = random.randint(0, self.ploidy - 1)
            a = self.genotype[na][g.name]
            gam[g.name] = a.mutate(rate = mutation_rate, p = mutation_p)
        
        return gam
    
    def randomize_alleles(self, **params):
        
        mean_scale = params.get("mean_scale", 1.0)
        sd_scale = params.get("sd_scale", 0.0)
        
        mean_threshold = params.get("mean_threshold", 0.5)
        sd_threshold = params.get("sd_threshold", 0.1)
        
        mean_decay = params.get("mean_decay", 0.05)
        sd_decay = params.get("sd_decay", 0.01)
        
        new_gnm = Genome(ploidy = self.ploidy)
        
        for gi in range(self.num_genes):
            gene = self.genes[gi]
            
            alleles = []
            
            for n in range(self.ploidy):
                scale = random.normalvariate(mean_scale, sd_scale)
                thr = random.normalvariate(mean_threshold, sd_threshold)
                decay = random.normalvariate(mean_decay, sd_decay)
                
                allele = gene.create_allele(f"Allele{n}", scale, thr, decay)
                alleles.append(allele)
            
            new_gnm.add_gene(gene, *alleles)
        
        inters = self.get_interaction_dict()
        
        for (ni, nj) in inters:
            gi = new_gnm.genes[ni]
            gj = new_gnm.genes[nj]
            
            wgt = inters[(ni, nj)]
            
            new_gnm.add_interaction(gi, gj, wgt)
        
        return new_gnm
    
    @classmethod
    def initialize_random(cls, num_genes, num_phenotypes, num_interactions, **params):
        
        ploidy = params.get("ploidy", 2)
        
        mean_cost = params.get("mean_cost", 1.0)
        sd_cost = params.get("sd_cost", 0.1)
        
        mean_exp = params.get("mean_expression", 0.5)
        sd_exp = params.get("sd_expression", 0.1)
        
        mean_scale = params.get("mean_scale", 1.0)
        sd_scale = params.get("sd_scale", 0.0)
        
        mean_threshold = params.get("mean_threshold", 0.5)
        sd_threshold = params.get("sd_threshold", 0.1)
        
        mean_decay = params.get("mean_decay", 0.05)
        sd_decay = params.get("sd_decay", 0.01)
        
        mean_wgt = params.get("mean_weight", 0.0)
        sd_wgt = params.get("sd_weight", 1.0)
        
        num_modules = params.get("num_modules", 1)
        sd_mod_frac = params.get("sd_module_fraction", 0.1)
        num_bridge = params.get("num_bridge", 0)
        
        gnm = cls(ploidy = ploidy)
        
        for g in range(num_genes):
            cost = random.normalvariate(mean_cost, sd_cost)
            gene = Gene(f"Gene{g}", cost_factor = cost)
            
            alleles = []
            
            for n in range(ploidy):
                scale = random.normalvariate(mean_scale, sd_scale)
                thr = random.normalvariate(mean_threshold, sd_threshold)
                decay = random.normalvariate(mean_decay, sd_decay)
                
                allele = gene.create_allele(f"Allele{n}", scale, thr, decay)
                alleles.append(allele)
                
            init_exp = random.normalvariate(mean_exp, sd_exp) 
            gnm.add_gene(gene, *alleles, expression = init_exp)
        
        max_inters = num_genes * (num_genes - 1)
        num_interactions = min(max_inters, num_interactions)
        # inter_pairs = list(itertools.chain.from_iterable([[(i, j + int(j==1)%gnm.num_genes) for i in range(num_genes)] for j in range(num_genes)])) # doopid
        inter_pairs = list(itertools.chain.from_iterable([[(i, j) for j in range(num_genes) if j!=i] for i in range(num_genes)]))
        inters = random.sample(inter_pairs, num_interactions)
        
        # inter_mods = []
        # if num_modules > 1:
            
        #     mod_f = 1 / num_modules
        #     mod_sizes = [num_genes*random.normalvariate(mod_f, sd_mod_frac) for n in range(num_modules)]
        #     err = num_genes - int(sum(mod_sizes))
        #     err_whole = err // num_modules
        #     err_res = err % num_modules
            
        #     i_last = 0
        #     for i in range(num_modules):
                
        #         i_mod = i_last + int(round(mod_sizes[i])) + err_whole
        #         if err_res > 0:
        #             i_mod +=1
        #             err_res -=1
                
        #         i_last = i_mod
        # else:
        #     inter_mods = [inters]
        
        for (ni, nj) in inters:
            gi = gnm.genes[ni]
            gj = gnm.genes[nj]
            
            wgt = random.normalvariate(mean_wgt, sd_wgt)
            
            gnm.add_interaction(gi, gj, wgt)

        for pg in range(num_phenotypes):
            
            cost = random.normalvariate(mean_cost, sd_cost)
            gene = PhenotypicGene(f"PhenotypicGene{g}", cost_factor = cost, min_value = 0.0, max_value = 1.0)
            
            for n in range(ploidy):
                
                scale = random.normalvariate(mean_scale, sd_scale)
                thr = random.normalvariate(mean_threshold, sd_threshold)
                decay = random.normalvariate(mean_decay, sd_decay)
                
                allele = gene.create_allele(f"PhenotypicAllele{n}", scale, thr, decay)
                alleles.append(allele)
                
        return gnm
    
    def __repr__(self):
        
        parts = []
        if self.ploidy == 1:
            pstr = "monoploid"
        if self.ploidy == 2:
            pstr = "diploid"
        if self.ploidy == 3:
            pstr = "triploid"
        if self.ploidy == 4:
            pstr = "tetraploid"
        parts.append(pstr)
        parts.append(f"{self.num_genes} genes")
        parts.append(f"{self.num_interactions} interactions")
        
        
        return f"{type(self).__name__}({", ".join(parts)})"




