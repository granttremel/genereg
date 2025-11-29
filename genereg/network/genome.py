

from typing import Dict, List, Tuple, Any, Union, Optional
import random
import string

import itertools

import numpy as np

import os
from pathlib import Path
import json

from genereg import draw
from genereg.network.gene import Gene, Allele, Phene, Phallele, Interaction
from genereg.utils import Rectifier, Epistasis



class Genome:
    """
    A genome represents the set of all genes that describe an organism and the interactions between them, i.e. a gene regulatory network.
    A gene is a node, connected to other genes by edges. each edge represents a gene's influence on another, including scale and sensitivity.
    The genome may be organized into modules, 
    Two genes should interact regardless of alleles..
    """
    
    def __init__(self, ploidy = 2, tag = ""):
        
        self.ploidy = ploidy
        self.tag = tag

        self.genes:Dict[str, Gene] = {}
        self.gene_order:List[str] = []  # Maintains insertion order for reliable indexing
        self.expression:Dict[str,float] = {}

        self.genotype:List[Dict[str, Allele]]= [{} for n in range(ploidy)]
        self.phenotype:Dict[str, float]

        self.interactions:Dict[str, Dict[str, Interaction]] = {}
    
    @property
    def num_genes(self):
        return len(self.genes)
    
    @property
    def num_phenes(self):
        return sum(1 for pi in self.iter_phenes())
    
    @property
    def num_interactions(self):
        return len(self.interactions)
    
    def count_upstream(self):
        
        num_upstream = []
        
        for gi in self.iter_genes():
            gjdict = self.interactions.get(gi.name, {})
            num_upstream.append(len(gjdict))
            
        for gi in self.iter_phenes():
            gjdict = self.interactions.get(gi.name, {})
            num_upstream.append(len(gjdict))
        
        return num_upstream
    
    def count_downstream(self):

        num_downstream = {gj_name:0 for gj_name in self.gene_order}

        for gi, gjdict in self.interactions.items():
            for gj, v in gjdict.items():
                num_downstream[gj] += 1

        return [num_downstream[gj_name] for gj_name in self.gene_order]
    
    def get_gene(self, gene_name, default = None):
        return self.genes.get(gene_name, default)
    
    def add_gene(self, gene:Gene, *alleles):
        self.genes[gene.name] = gene
        self.gene_order.append(gene.name)
        self.expression[gene.name] = 0.0
        for i in range(self.ploidy):
            self.genotype[i][gene.name] = alleles[i]
        if gene.is_phenotype:
            self.interactions[gene.name] = {}
        else:
            self.interactions[gene.name] = {}
            
    def set_allele(self, allele:Allele, nchr):
        self.genotype[nchr][allele.gene.name] = allele
    
    def make_interaction(self, effector_gene_ind,  affected_gene_ind, weight):
        g_aff = self.get_gene(affected_gene_ind)
        g_eff = self.get_gene(effector_gene_ind)
        inter = Interaction(g_eff, g_aff, weight = weight)
        
        self.add_interaction(inter)
    
    def add_interaction(self, new_inter:Interaction):
        
        if not isinstance(new_inter, Interaction):
            raise ValueError(f"{new_inter} is not valid Interaction")
        
        gn_aff = new_inter.affected.name
        gn_eff = new_inter.effector.name
        
        if not gn_aff in self.interactions:
            self.interactions[gn_aff] = {}
        
        self.interactions[gn_aff][gn_eff] = new_inter
        
    def add_interactions(self, new_inters:Dict[Tuple[str,str], Interaction]):
        
        for (e, a), inter in new_inters.items():
            self.add_interaction(inter)
    
    def get_expression(self):
        return np.array([self.expression[g.name] for g in self])
    
    def get_product(self):
        return np.vstack([np.array([self.genotype[chr][g_name].product for g_name in self.gene_order]) for chr in range(self.ploidy)])
    
    def get_interactions(self):

        inters = np.zeros((self.num_genes, self.num_genes))

        for g1, g0dict in self.interactions.items():
            for g0, v in g0dict.items():
                g0ind = self.gene_order.index(g0)
                g1ind = self.gene_order.index(g1)
                inters[g0ind, g1ind] = v.weight

        return inters
    
    def get_interaction_dict(self):
        inter_dict = {}

        for g1, g0dict in self.interactions.items():
            for g0, v in g0dict.items():
                g0ind = self.gene_order.index(g0)
                g1ind = self.gene_order.index(g1)
                inter_dict[(g0ind, g1ind)] = v.weight

        return inter_dict
    
    def get_state(self):

        gene_states = {}
        allele_states = {n:{} for n in range(self.ploidy)}
        interaction_states = {}

        for gene_name in self.gene_order:
            gene = self.genes[gene_name]
            gene_states[gene.name] = gene.get_state()
            for n in range(self.ploidy):
                allele = self.genotype[n][gene_name]
                allele_states[n][gene_name] = allele.get_state()

            interaction_states[gene_name] = {}
            for gj, inter in self.interactions.get(gene_name, {}).items():
                interaction_states[gene_name][gj] = inter.get_state()

        return {
            "genes":gene_states,
            "genotype":allele_states,
            "interactions":interaction_states,
            "gene_order":self.gene_order
        }
    
    def set_state(self, state_dict):

        genes = state_dict.get("genes",[])
        genotype = state_dict.get("genotype",[])
        interactions = state_dict.get("interactions",{})

        for gn, gd in genes.items():
            gene = self.get_gene(gn)
            gene.set_state(gd)

            for n in range(self.ploidy):
                ad = genotype[n][gene.name]
                allele = self.genotype[n][gene.name]
                allele.set_state(ad)

            for gid, interd in interactions.get(gene.name, {}).items():

                if gid in self.interactions[gene.name]:
                    inter = self.interactions[gene.name][gid]
                    if hasattr(inter, 'set_state'):
                        inter.set_state(interd)
                else:
                    g_eff = self.genes.get(gid)
                    if g_eff and hasattr(Interaction, 'from_state'):
                        new_inter = Interaction.from_state(interd, g_eff, gene)
                        self.interactions[gene.name][gid] = new_inter
                
    def save_state(self, fname, **metadata):
        state = self.get_state()
        state["meta"] = metadata
        fpath = os.path.join("./data", fname)
        with open(fpath, "w+") as f:
            json.dump(state, f, indent = 3)
    
    def update_expression(self, resource = -1):

        new_exprs = {}
        
        for gi in self:
            
            inters_i = self.interactions[gi.name]

            # Calculate the weighted sum of interactions once (same for all chromosomes)
            # sum w_ij * a_j, w is interaction, a is expression
            regulation = np.sum([inter_ij.evaluate() for gj, inter_ij in inters_i.items()])
            gi.regulation = regulation

            # Evaluate each chromosome's allele independently
            allele_products = []
            for chr in range(self.ploidy):
                allele = self.genotype[chr][gi.name]
                allele_product = allele.evaluate(regulation)
                allele_products.append(allele_product)
            
            # Combine allele products using the gene's epistasis function
            new_exprs[gi.name] = gi.evaluate(*allele_products)
        
        if resource > 0:
            sum_expr = sum(new_exprs.values())
            if sum_expr > resource:
                scale = resource / sum_expr
                for g in self:
                    g.scale_expression(scale)
                    new_exprs[g.name] = g.gene_product

        self.expression = new_exprs
        return self.get_expression()
    
    def initialize(self, init_product = None, **kwargs):
        
        init_exprs = {}
        
        if init_product:
            for chr in self.genotype:
                for gn, a in chr.items():
                    # a.initialize(init_product.pop(0))
                    a.initialize(init_product())
        
        for gene in self:
            aps = [self.genotype[n][gene.name].product for n in range(self.ploidy)]
            try:
                gp = gene.evaluate(*aps)
            except:
                print(aps, gene.name, gene.alleles)
            init_exprs[gene.name] = gp
        self.expression = init_exprs
    
    def silence_gene(self, nchr, gene_name):
        allele = self.genotype[nchr][gene_name]
        
        if isinstance(allele, Phallele):
            # at least one should be active or the organism perishes
            pass
        
        allele.toggle_silence()
    
    def unsilence_gene(self, nchr, gene_name):        
        allele = self.genotype[nchr][gene_name]
        
        if isinstance(allele, Phallele):
            # at least one should be active or the organism perishes
            pass
        
        allele.toggle_silence(is_silent = False)
    
    def get_gamete(self, mutation_rate = 0.1, mutation_p = 0.2):

        gam = {}

        for gene_name in self.gene_order:
            na = random.randint(0, self.ploidy - 1)
            a = self.genotype[na][gene_name]
            new_allele = a.mutate(rate = mutation_rate, p = mutation_p)
            
            gam[gene_name] = new_allele

        return gam
    
    def print(self):
        draw.show_genome(self)
    
    def show_interactions(self):
        draw.show_interactions(self)
    
    def show_interaction_heatmap(self, **kwargs):
        draw.show_interaction_heatmap(
            self.get_interactions(), 
            row_labels = self.gene_order,
            **kwargs)
        
    
    def with_mutant_alleles(self, rate, p, **params):
        
        new_gnm = Genome(ploidy = self.ploidy)

        for gene in self:

            alleles = []

            for n in range(self.ploidy):

                new_allele = self.genotype[n][gene.name].mutate(rate, p)
                alleles.append(new_allele)

            new_gnm.add_gene(gene, *alleles)

        for gn_aff in self.interactions:
            for gn_eff, inter in self.interactions[gn_aff].items():
                new_inter = inter.mutate(rate, p)
                new_gnm.add_interaction(new_inter)
        
        return new_gnm
    
    # def randomize_alleles(self, **params):

    #     mean_scale = params.get("mean_scale", 1.0)
    #     sd_scale = params.get("sd_scale", 0.0)

    #     mean_threshold = params.get("mean_threshold", 0.5)
    #     sd_threshold = params.get("sd_threshold", 0.1)

    #     mean_decay = params.get("mean_decay", 0.05)
    #     sd_decay = params.get("sd_decay", 0.01)

    #     new_gnm = Genome(ploidy = self.ploidy)

    #     for gene in self:

    #         alleles = []

    #         for n in range(self.ploidy):
    #             scale = random.normalvariate(mean_scale, sd_scale)
    #             thr = random.normalvariate(mean_threshold, sd_threshold)
    #             decay = random.normalvariate(mean_decay, sd_decay)

    #             allele = gene.create_allele(f"Allele{n}", scale, thr, decay)
    #             alleles.append(allele)

    #         new_gnm.add_gene(gene, *alleles)

    #     inters = self.get_interaction_dict()

    #     for (ni, nj) in inters:
    #         wgt = inters[(ni, nj)]

    #         new_gnm.make_interaction(ni, nj, wgt)

    #     return new_gnm
        
    def get_gene(self, ind):
        gene = None
        if isinstance(ind, int):
            gn = self.gene_order[ind]
            gene = self.genes.get(gn)
        elif isinstance(ind, str):
            gene = self.genes.get(ind)
        
        if gene is None:
            raise ValueError(f"Index {ind} not valid")
        
        return gene
    
    def __getitem__(self, ind):
        return self.get_gene(ind)
    
    def iter_genes(self):
        for gn in self.gene_order:
            g = self.genes[gn]
            if g.is_genotype:
                yield g
                
    def iter_phenes(self):
        for gn in self.gene_order:
            g = self.genes[gn]
            if g.is_phenotype:
                yield g

    def iter_bridges(self):
        for gn in self.order:
            g = self.get_gene(gn)
            if g.is_bridge:
                yield g

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


class CompiledGenome:
    
    def __init__(self, ploidy, num_genes, **kwargs):
    
        self.ploidy = ploidy
        self.num_genes = num_genes
    
        self.allele_scale = np.zeros((num_genes, ploidy))
        self.allele_threshold = np.zeros((num_genes, ploidy))
        self.allele_decay = np.zeros((num_genes, ploidy))
        
        rect_type = kwargs.get("rect","Sigmoid")
        self.gene_rect = Rectifier(rect_type)
        epi_type = kwargs.get("epistasis","sum")
        self.gene_epistasis = Epistasis(epi_type)
        
        self.interactions = np.zeros((num_genes, num_genes))
        
        self.reg_state = np.zeros((num_genes))
        self.allele_state = np.zeros((num_genes, ploidy))
        self.gene_state = np.zeros((num_genes))
    
    def initialize(self, mean_init_prod, sd_init_prod):
        self.allele_state = np.random.normal(mean_init_prod, sd_init_prod, size = (self.num_genes, self.ploidy))
    
    def update(self):
        rg, ast, exp = self.calculate_gene_state(self.gene_state, self.allele_state)
        self.reg_state = rg
        self.allele_state = ast
        self.gene_state = exp
        return rg, ast, exp
        
    def calculate_gene_state(self, expression, allele_state):
        
        regulation = np.matvec(self.interactions, expression)
        val = np.subtract(regulation, self.allele_threshold, axis = 0)
        val = np.multiply(val, self.allele_scale, axis = 0) 
        new_allele_state = val + np.multiply(allele_state, self.allele_decay, axis = 0)
        
        new_expression = self.gene_epistasis.func(new_allele_state, axis = 1)
        
        return regulation, new_allele_state, new_expression
        
    @classmethod
    def from_genome(cls, gnm:Genome):
        
        ploidy = gnm.ploidy
        num_genes = len(gnm)
        
        allele_scale = np.zeros((num_genes, ploidy))
        allele_threshold = np.zeros((num_genes, ploidy))
        allele_decay = np.zeros((num_genes, ploidy))
        
        gene_rect = gnm[0].rect
        gene_epistasis = gnm[0].epistasis
        
        interactions = np.zeros((num_genes, num_genes))
        
        for i,gene in enumerate(gnm):
            for n in range(gnm.ploidy):
                
                allele = gnm.genotype[n][gene.name]
                allele_scale[i,n] = allele.scale
                allele_threshold[i,n] = allele.threshold
                allele_decay[i,n] = allele.decay
                
            for gj, inter in gnm.interactions.get(gene.name,{}).items():
                j = gnm.gene_order.index(gj)
                interactions[gj, i] = inter.weight
        
        return cls(ploidy, num_genes,
                allele_scale = allele_scale,
                allele_threshold = allele_threshold,
                allele_decay = allele_decay,
                rect = gene_rect,
                epistasis = gene_epistasis,
            )
        
        
        
        
    
    

