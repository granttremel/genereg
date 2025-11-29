
from typing import Dict, List, Tuple, Any, Union, Optional
from tabulate import tabulate
import random

from scipy.stats import poisson
import numpy as np

from scipy.stats import differential_entropy
from scipy.spatial.distance import cosine

from genereg.network import gene
from genereg.network.genome import Genome
from genereg.network.gene import Gene, Allele, Phene, Phallele, Interaction

from genereg import draw

class Organism:
    """
    An organism with a genome. the genome determines the parameterization of the organism, and the organism determines the fitness of the genome. Organisms are the site of classical natural selection, in this context acting as an optimization algorithm.
    """
    
    cols = [53, 136]
    
    def __init__(self, genome:Genome, resource = -1):
        
        self.genome = genome
        self.resource = resource
        self.history = {"expression":[], "product":[]}
        
        self.quantity = 0
        self.t = 0
        
    def initialize(self, init_product = None):
        if init_product is None:
            init_product = []
        self.genome.initialize(init_product=init_product)
        self.record_state(self.genome.get_expression(), self.genome.get_product())
        
    def step(self):
        
        new_expr  = self.genome.update_expression(resource = self.resource)
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
            for gi, g in enumerate(self.genome):
                v = self.history["expression"][tt][gi]
                data[tt, gi] = v
        
        return data
    
    def show_history(self, show_product = False):
        draw.show_history(self.genome, self.history, self.t, show_product = show_product)
        
    def plot_expression(self, genes = False, phenes = True):
        self.plot_expressions(self, genes=genes, phenes=phenes)
        
    @classmethod
    def plot_interesting_genes(cls, *orgs, topk = 5, headers = []):
        draw.plot_interesting_genes(*orgs, topk = topk, headers = headers)
    
    @classmethod
    def plot_expressions(cls, *orgs, genes = False, phenes = True, gene_names = []):
        draw.plot_expressions(*orgs, genes=genes, phenes = phenes, gene_names = gene_names)
            
    def plot_product(self):
        draw.plot_product(self.genome, self.history, self.t)
    
    def plot_states(self):
        draw.plot_states(self.genome, self.history)
    
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
            
            datai = time_data[:, gi]
            entr = differential_entropy(datai)
            if np.isfinite(entr):
                interestingness += entr
                ni += 1
            cov_mat = np.cov(time_data)
            uniqueness = np.sum(cov_mat.diagonal() / np.sum(np.abs(cov_mat / cov_mat.diagonal())))
            # for gj in range(self.genome.num_genes):
                
            #     dataj = time_data[:, gj]
            #     cov_mat = np.cov(datai, dataj)
            #     cov = cov_mat[0,1]
            #     var = np.sqrt(cov_mat[0,0]*cov_mat[1,1])
            #     if var > 0:
            #         uniqueness += 1 - abs(cov/var)
        
        
        if ni > 0:
            interestingness /= ni
        uniqueness /= self.genome.num_genes**2
        
        return interestingness, uniqueness
    
    def reproduce(self, other:'Organism', num_offspring, mutation_rate = 0.1, mutation_p = 0.2):
        
        offspring = []
        
        for n in range(num_offspring):
            gam0 = self.genome.get_gamete(mutation_rate = mutation_rate, mutation_p = mutation_p)
            gam1 = other.genome.get_gamete(mutation_rate = mutation_rate, mutation_p = mutation_p)
            
            new_genome = Genome(ploidy = self.genome.ploidy)
            
            for gene_name in self.genome.gene_order:
                g = self.genome.genes[gene_name]
                new_genome.add_gene(g, gam0[gene_name], gam1[gene_name])
            
            for (i, j), v in self.genome.get_interaction_dict().items():
                new_genome.make_interaction(i, j, v)
            
            new_org = Organism(new_genome)
            offspring.append(new_org)
        
        return offspring
    

class Population:
    
    def __init__(self, base_genome):
        
        self.base_genome = base_genome
        self.individuals:List[Organism]= []
        self.gen = 0
    
    @property
    def pop_size(self):
        return len(self.individuals)
    
    def add_individual(self, ind):
        self.individuals.append(ind)
    
    def initialize(self, init_product = None):
        if init_product is None:
            init_product = []
        for ind in self.individuals:
            ind.initialize(init_product=init_product)
    
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
            indi.quantity = qt
            orgranks.append(indi)
        
        orgranks = sorted(orgranks, key = lambda k:-k.quantity)
        topk_orgs = orgranks[:topk]
        return topk_orgs, [org.quantity for org in topk_orgs]
    
    def step_generation(self, surviving:List[Organism], mean_offspring, mutation_rate = 0.1, mutation_p = 0.2):
        
        random.shuffle(surviving)
        
        if len(surviving) % 2 == 1:
            surviving = surviving[:-1]
        
        qts = [org.quantity for org in surviving]
        mean_qt = np.mean(qts)
        sd_qt = np.std(qts)
        
        all_offs = []
        
        for i in range(0, len(surviving), 2):
            ind1 = surviving[i]
            ind2 = surviving[i+1]
            
            net_qt = max(ind1.quantity, ind2.quantity)
            qt_z = (net_qt - mean_qt) / sd_qt
            
            mr = mutation_rate * (np.exp(-qt_z))
            mp = mutation_p * (np.exp(-qt_z))
            
            noffs = poisson.rvs(mean_offspring)
            new_offs = ind1.reproduce(ind2, noffs, mutation_rate = mr, mutation_p = mp)
            all_offs.extend(new_offs)
        
        random.shuffle(all_offs)
        self.individuals = all_offs
        self.gen += 1
        
        return all_offs
    
    def step_epoch(self, epoch_size, mutation_rate, mutation_p, **kwargs):
        
        mean_offs = kwargs.pop("mean_offs", 2)
        
        num_timesteps = kwargs.pop("num_timesteps", 20)
        init_product = kwargs.pop("init_product", [])
        suppress = kwargs.pop("suppress", False)
        topk_final = kwargs.pop("topk_final", 8)
        
        init_pop_size = self.pop_size
        
        if not suppress:
            print(f"starting epoch size {epoch_size} with initial population {self.pop_size}")
        
        for n in range(epoch_size):
            self.initialize(init_product = init_product)
            self.step_to(num_timesteps)
            
            curr_mean_offs = mean_offs * init_pop_size / self.pop_size
            n_parents = int(2 * init_pop_size / curr_mean_offs)
            
            extra = 0
            
            topk_orgs, topk_uniqs = self.quantify(topk = n_parents)
            
            new_offs = self.step_generation(topk_orgs, curr_mean_offs + extra, mutation_rate = mutation_rate, mutation_p = mutation_p)
            if not suppress:
                print(f"completed gen {self.gen}, with pop size {self.pop_size} from mean parity {curr_mean_offs:0.3f}, mean uniqueness {np.mean(topk_uniqs):0.3e}")
        
        self.initialize(init_product = init_product)
        self.step_to(num_timesteps)
        topk_orgs, topk_uniqs = self.quantify(topk = topk_final)
        return topk_orgs
        
        
    def show_top(self, topk, metric = "uniqueness", genes = False, gene_names = []):
        print(f"most unique organism (gen {self.gen})")
        use_uniqueness = metric == "uniqueness"
        
        topk_orgs, topk_mets = self.quantify(topk = topk, use_uniqueness = use_uniqueness)
        
        headers = [f"Indi{n}({topk_mets[n]:0.3f})" for n in range(len(topk_mets))]
        
        topk_orgs[0].show_genome()
        Organism.plot_expressions(*topk_orgs, genes = genes, headers = headers, gene_names = gene_names)
        
        return topk_orgs
            
    
        
class Quantifier:
    
    def __init__(self):
        
        self.metrics = {"intensity":self.get_intensity,
                        "interestingness":self.get_interestingness,
                        "excitingness":self.get_excitingness}
        self.group_metrics = {"uniqueness", self.get_uniqueness}
        
    def get_metrics(self, time_series, metrics):
        
        mets = np.zeros(len(metrics))
        
        for i,met in enumerate(metrics):
            if met in self.metrics:
                val = self.metrics[met](time_series)
                mets[i] = val
                
        return mets
    
    def get_group_metrics(self, grp_time_series, grp_metrics):
        
        mets = np.zeros((len(grp_time_series, len(grp_metrics))))
        
        for i,met in enumerate(grp_metrics):
            if met in self.group_metrics:
                val = self.group_metrics[met](grp_time_series)
                mets[i, :] = val
            
            if met in self.metrics:
                for j in range(len(grp_time_series)):
                    val = self.metrics[met](grp_time_series[j])
                    mets[i, j] = val
                
        return mets
        
    
    def get_intensity(self, time_series):
        return np.mean(time_series, axis = 1)
    
    def get_interestingness(self, time_series):
        return np.std(time_series, axis = 1)
    
    def get_excitingness(self, time_series):
        
        entr = differential_entropy(time_series)
        if not np.isfinite(entr):
            entr = 0.0
            
        return entr
    
    def get_uniqueness(self, grp_time_series):
        
        cov_mat = np.cov(grp_time_series)
        uniqueness = cov_mat.diagonal / np.sum(np.abs(np.divide(cov_mat, cov_mat.diagonal(), axis = 1)))
        return uniqueness