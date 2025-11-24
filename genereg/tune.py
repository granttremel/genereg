
import random

from genereg.network.genome import Genome


class Tuner:
    
    def __init__(self, genome):
        self.genome:Genome = genome
        
        st, th = self.genome.get_state()
        self.init_state = st
        self.init_theta = th
        
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
        




