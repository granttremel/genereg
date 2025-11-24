
from typing import Dict, List, Tuple, Any, Union, Optional
from tabulate import tabulate
import random

from scipy.stats import poisson
import numpy as np

from scipy.stats import differential_entropy
from scipy.spatial.distance import cosine

from genereg.network import gene
from genereg.network.genome import Genome
from genereg.network.gene import Gene, Allele, PhenotypicGene, PhenotypicAllele

from genereg import draw

class Organism:
    """
    An organism with a genome. the genome determines the parameterization of the organism, and the organism determines the fitness of the genome. Organisms are the site of classical natural selection, in this context acting as an optimization algorithm.
    """
    
    def __init__(self, genome:Genome):
        
        self.genome = genome
        self.history = {"expression":[], "product":[]}
        
        self.t = 0
        
    def initialize(self, init_product = None, mean_init_product = 0.1, sd_init_product = 0.02):
        if init_product is None:
            init_product = []
        self.genome.initialize(init_product=init_product, mean_init_product=mean_init_product, sd_init_product=sd_init_product)
        self.record_state(self.genome.get_expression(), self.genome.get_product())
        
    def step(self):
        
        new_expr  = self.genome.update_expression()
        new_prod = self.genome.get_product()
        
        self.record_state(new_expr, new_prod)
        self.t+=1
    
    def step_to(self, num_steps):
        for tt in range(self.t, num_steps):
            self.step()
    
    def record_state(self, expression, product):
        self.history["expression"].append(expression)
        self.history["product"].append(product)
    
    def get_time_series(self):
        
        data = np.zeros((self.t, self.genome.num_genes))
        
        for tt in range(self.t):
            for gi in range(self.genome.num_genes):
                v = self.history["expression"][tt][gi]
                data[tt, gi] = v
        
        return data
    
    def show_history(self, show_product = False):
        
        hdrs = ["Gene"]
        rows = [[self.genome.genes[ng].name] for ng in range(self.genome.num_genes)]
        
        for tt in range(self.t+1):
            hdrs.append(f"Expr(t={tt})")
            if show_product:
                hdrs.extend([f"Prod(t={tt}, chr={n})" for n in range(self.genome.ploidy)])
            
            expr = self.history["expression"][tt]
            prod = self.history["product"][tt]
            for ng in range(self.genome.num_genes):
                rows[ng].append(expr[ng])
                if show_product:
                    for nch in range(self.genome.ploidy):
                        rows[ng].append(prod[nch][ng])
        
        print(tabulate(rows, headers = hdrs, floatfmt = "0.3f"))
        print()
        
    def plot_expression(self):
        
        rowfmt = "{:<8}{}"
        
        for ng in range(self.genome.num_genes):
            
            gene_data = []
            for tt in range(self.t+1):
                expr = self.history["expression"][tt][ng]
                gene_data.append(expr)
            
            sctxt = draw.scalar_to_text_nb(gene_data, minval = 0, add_range = True)
            lbls = [self.genome.genes[ng].name, "", ""]
            for r, lbl in zip(sctxt, lbls):
                print(rowfmt.format(lbl, r))
            print()
            
    @classmethod
    def plot_expressions(cls, *orgs):
        
        rowfmt = "{:<8}{}"
        
        for ng in range(orgs[0].genome.num_genes):
            rows = [[],[],[]]
            for org in orgs:
                
                gene_data = []
                for tt in range(org.t+1):
                    expr = org.history["expression"][tt][ng]
                    gene_data.append(expr)
                
                sctxt = draw.scalar_to_text_nb(gene_data, minval = 0, add_range = True)
                for i in range(len(sctxt)):
                    rows[i].append(sctxt[i])
            
            lbls = [org.genome.genes[ng].name, "", ""]
            for lbl,row in zip(lbls,rows):
                print(rowfmt.format(lbl, "    ".join(row)))
            print()
            
    
    def plot_product(self):
        
        rowfmt = "{:<8}{}"
        
        for ng in range(self.genome.num_genes):
            print(self.genome.genes[ng].name)
            for nchr in range(self.genome.ploidy):
                gene_data = []
                for tt in range(self.t+1):
                    prod = self.history["product"][tt][nchr][ng]
                    gene_data.append(prod)
            
                sctxt = draw.scalar_to_text_nb(gene_data, minval = 0, add_range = True)
                lbls = [self.genome.genotype[nchr][self.genome.genes[ng].name].id, "", ""]
                for r, lbl in zip(sctxt, lbls):
                    print(rowfmt.format(lbl, r))
                print()
    
    def show_genome(self):
        
        hdrs = ["Gene", "Expression","# Downstream", "# Upstream", "Product", "Scale", "Threshold", "Decay"]
        rows = []
        
        num_upstream = self.genome.count_upstream()
        num_downstream = self.genome.count_downstream()
        
        rows = []
        for gi in range(self.genome.num_genes):
            g = self.genome.genes[gi]
            
            row = [g.name, format(self.genome.expression[g.name], "0.3f")]
            row.extend([num_downstream[gi], num_upstream[gi]])
            row.extend(["--" for i in range(4)])
            rows.append(row)
            
            for nch in range(self.genome.ploidy):
                a = self.genome.genotype[nch][g.name]
                row = [a.id, "--", "--", "--", format(a.product,"0.3f"), format(a.scale,"0.3f"), format(a.threshold,"0.3f"), format(a.decay,"0.3f")]
                rows.append(row)
        
        print(tabulate(rows, headers = hdrs, floatfmt = "0.3f"))
        print()
    
    def detect_stop(self, time_data, thresh = 1e-4):
        
        for tt in range(1, self.t):
            delta = cosine(time_data[tt], time_data[tt-1])
            if delta < thresh:
                return tt
        return self.t
    
    def quantify(self):
        
        ni = 0
        interestingness = 0
        uniqueness = 0
        
        time_data = self.get_time_series()
        
        opt_cov = 1/np.sqrt(self.genome.num_genes)
        
        for gi in range(self.genome.num_genes):
            
            datai = time_data[gi]
            entr = differential_entropy(datai)
            if np.isfinite(entr):
                interestingness += entr
                ni += 1
            
            for gj in range(self.genome.num_genes):
                
                dataj = time_data[gj]
                cov_mat = np.cov(datai, dataj)
                cov = cov_mat[0,1]
                var = np.sqrt(cov_mat[0,0]*cov_mat[1,1])
                if var > 0:
                    uniqueness += 1 - abs(cov/var)
        
        if ni > 0:
            interestingness /= ni
        uniqueness /= self.genome.num_genes **2
        
        return interestingness, uniqueness
    
    def reproduce(self, other:'Organism', num_offspring, mutation_rate = 0.1, mutation_p = 0.2, mean_expression = 0.0, sd_expression = 0.0):
        
        offspring = []
        
        for n in range(num_offspring):
            gam0 = self.genome.get_gamete(mutation_rate = mutation_rate, mutation_p = mutation_p)
            gam1 = other.genome.get_gamete(mutation_rate = mutation_rate, mutation_p = mutation_p)
            
            new_genome = Genome(ploidy = self.genome.ploidy)
            
            for g in self.genome.genes:
                expr = random.normalvariate(mean_expression, sd_expression)
                new_genome.add_gene(g, gam0[g.name], gam1[g.name], expression = expr)
            
            for (i, j), v in self.genome.get_interaction_dict().items():
                gi = self.genome.genes[i]
                gj = self.genome.genes[j]
                new_genome.add_interaction(gj, gi, v)
            
            new_org = Organism(new_genome)
            offspring.append(new_org)
        
        return offspring
    

class Population:
    
    def __init__(self, base_genome):
        
        self.base_genome = base_genome
        self.individuals:List[Organism]= []
        
    def add_individual(self, ind):
        self.individuals.append(ind)
    
    def initialize(self, init_product = None, mean_init_product = 0.1, sd_init_product = 0.02):
        if init_product is None:
            init_product = []
        for ind in self.individuals:
            ind.initialize(init_product=init_product, mean_init_product=mean_init_product, sd_init_product=sd_init_product)
    
    def step_to(self, t):
        for ind in self.individuals:
            ind.step_to(t)
            
    def quantify(self, topk = 8, use_uniqueness = True):
        
        orgranks = []
        
        for indi in self.individuals:
            inch, uniq = indi.quantify()
            if use_uniqueness:
                qt = uniq
            else:
                qt = inch
            orgranks.append((indi, qt))
        
        orgranks = sorted(orgranks, key = lambda k:-k[1])
        topk_orgs = orgranks[:topk]
        return [org for org, qt in topk_orgs], [qt for org, qt in topk_orgs]
    
    def step_generation(self, surviving, mean_offspring, mutation_rate = 0.1, mutation_p = 0.2, mean_expression = 0.0, sd_expression = 0.0):
        
        random.shuffle(surviving)
        
        if len(surviving) % 2 == 1:
            surviving = surviving[:-1]
        
        all_offs = []
        
        for i in range(0, len(surviving), 2):
            ind1 = surviving[i]
            ind2 = surviving[i+1]
            noffs = poisson.rvs(mean_offspring)
            new_offs = ind1.reproduce(ind2, noffs, mutation_rate = mutation_rate, mutation_p = mutation_p, mean_expression = mean_expression, sd_expression=sd_expression)
            all_offs.extend(new_offs)
        
        random.shuffle(all_offs)
        self.individuals = all_offs
        return all_offs