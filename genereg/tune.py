
from typing import Dict, List, Any
from dataclasses import dataclass

import json

import numpy as np
import random

from genereg.network.genome import Genome
from genereg.organism import Organism

@dataclass
class State:
    state: np.ndarray
    theta: np.ndarray

    @property
    def num_state_params(self):
        return len(self.state)
    
    @property
    def num_thetas(self):
        return len(self.theta)

class Tuner:
    
    def __init__(self, genome):
        self.genome:Genome = genome
        
        self.state_map = []
        self.theta_map = []
        
        self.states:Dict[str, State] = {}
        
        self.save_state(state_key = "InitState")
        self.init_state:State = self.states.get("InitState")
    
    def save_state(self, state_key = ""):
        
        state = self.genome.get_state()
        
        if not self.state_map:
            self.initialize_state_maps(state)
        
        state_vec = self.state_dict_to_vector(state, self.state_map)
        theta_vec = self.state_dict_to_vector(state, self.theta_map)
        
        if not state_key:
            state_key = f"State{len(self.states)}"
        
        self.states[state_key] = State(state_vec, theta_vec)
        return self.states[state_key]
    
    def get_state(self, state_key):
        return self.states.get(state_key)
    
    def get_state_dict(self, state_key):
        state = self.states.get(state_key)
        sdict = self.state_vector_to_dict(state.state, self.state_map)
        thdict = self.state_vector_to_dict(state.theta, self.theta_map)
        return sdict, thdict
    
    def state_vector_to_dict(self, state_vec, map):
        
        state_dict = {}
        
        for i in range(len(map)):
            
            stkey = map[i]
            
            _st = state_dict
            for stk in stkey[:-1]:
                if not stk in _st:
                    _st[stk] = {}
                _st = _st[stk]
            _st[stkey[-1]] = state_vec[i]
            
        return state_dict
    
    def state_dict_to_vector(self, state_dict, map):
        
        state_vec = np.zeros((len(map)))
        
        for i in range(len(map)):
            stkey = map[i]
            
            _st = state_dict
            for stk in stkey:
                _st = _st[stk]
            
            if isinstance(_st, float):
                state_vec[i] = _st
            else:
                state_vec[i] = 0.0
        
        return state_vec
    
    def initialize_state_maps(self, state):
        
        state_map = []
        theta_map = []
        
        order = state.get("gene_order",[])
        genes = state.get("genes",{})
        genotype = state.get("genotype",{})
        interactions = state.get("interactions",{})
        
        for gn in order:
            gd = genes.get(gn)
            if not gd:
                continue
            
            state_map.append(("genes", gn, "gene_product"))
            state_map.append(("genes", gn, "wijaj"))
            theta_map.append(("genes", gn, "cost_factor"))
        
        for nchr, chrdict in genotype.items():
            
            for gn in order:
                allele = chrdict.get(gn)
                if not allele:
                    continue
                
                state_map.append(("genotype",nchr, gn, "product"))
                for att in ["scale","threshold","decay"]:
                    theta_map.append(("genotype", nchr, gn, att))
        
        for gni in order:
            gnd = interactions.get(gni, {})
            for gnj, inter in gnd.items():
                for att in ["weight"]:
                    theta_map.append(("interactions", gni, gnj, att))
                
        self.state_map = state_map
        self.theta_map = theta_map
    
    def evaluate_genome(self, t_max, init_conds = None, thetas = None, state_key = ""):
        
        if init_conds is not None:
            if isinstance(init_conds, np.ndarray):
                init_conds = self.state_vector_to_dict(init_conds, self.state_map)
            self.genome.set_state(init_conds)
            
        if thetas is not None:
            if isinstance(thetas, np.ndarray):
                thetas = self.state_vector_to_dict(thetas, self.theta_map)
            self.genome.set_state(thetas)
        
        org = Organism(self.genome)
        org.step_to(t_max)
        
        new_state = self.save_state(state_key = state_key)
        
        return new_state
    
    def measure_noise_sensitivity(self, noise_sigma, num_tests = 10, t_max = 20):
        
        init_conds = self.state_vector_to_dict(self.init_state.state, self.state_map)
        self.genome.set_state(init_conds)
        init_res = self.evaluate_genome(t_max, init_conds = init_conds)
        
        dstates = []
        
        for n in range(num_tests):
            
            ttheta = self.init_state.theta + np.random.normal(0, noise_sigma, size = self.init_state.num_thetas)
            new_state = self.evaluate_genome(t_max, init_conds = init_conds, thetas = ttheta, state_key = f"Test{n}")
            
            dstate = new_state.state - init_res.state
            dstates.append(dstate)
        
        dstates = np.array(dstates)
        state_sigs = np.std(dstates, axis = 0)
        state_gains = state_sigs/noise_sigma
        state_gain_dict = self.state_vector_to_dict(state_gains, self.state_map)
        
        return state_gain_dict
    
    def calculate_sensitivity(self, t_meas, rate = 0.02):
        
        self.genome.initialize()
        for i in range(t_meas):
            self.genome.update_expression()
        
        state, theta = self.genome.get_state()
        
        dtheta = [th + rate for th in theta]
        
        self.genome.set_state(self.init_state, dtheta)
        
        self.genome.initialize()
        for i in range(t_meas):
            self.genome.update_expression()
        
        fstate, ftheta = self.genome.get_state()
        
        dstate = [fst - st for st, fst in zip(state, fstate)]
        dstatedtheta = [dst/rate for dst in dstate]
        
        return dstatedtheta
        




