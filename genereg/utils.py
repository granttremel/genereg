

import numpy as np

class Rectifier:
    
    _rect_params = {
        "Identity":[],
        "Tanh":[],
        "Sigmoid":[],
        "ReLU":[],
        "LeakyReLU":["leak"],
    }
    
    _rect_range = {
        "Identity":[None, None],
        "Tanh":[-1.0, 1.0],
        "Sigmoid":[0.0, 1.0],
        "ReLU":[0.0, None],
        "LeakyReLU":[None, None],
    }
    
    def __init__(self, rect_func, **params):
        self.func_type = rect_func
        self.func = self.get_func(rect_func)
        self._params = params
    
    def __call__(self, value):
        return self.apply(value)
    
    def apply(self, value):
        return self.func(value)
    
    def apply_rescale(self, value, min_value, max_value):
        
        raw_value = self.apply(value)
        bds = self._rect_range.get(self.func_type)
        
        if None in bds:
            raise ValueError
        
        out_value = (raw_value + bds[0])*(bds[1]-bds[0]) * (max_value - min_value) + min_value
        return out_value
    
    def _identity(self, value):
        return value
        
    def _tanh(self, value):
        return np.tanh(value)
    
    def _sigmoid(self, value):
        return 1/(1+np.exp(-value))
    
    def _relu(self, value):
        return value if value > 0.0 else 0.0
    
    def _lrelu(self, value):
        return value if value > 0.0 else value * self._params[0]
    
    def get_func(self, rect_func):
        
        if rect_func == "Tanh":
            return self._tanh
        elif rect_func == "Sigmoid":
            return self._sigmoid
        elif rect_func == "ReLU":
            return self._relu
        elif rect_func == "LeakyReLU":
            return self._lrelu
        else:
            return self._identity
        
    def get_params(self):
        return self._params
    
    def update_params(self, param_list):
        self._params = param_list
        return self._params
    
    @property
    def num_params(self):
        return len(self._rect_params.get(self.func_type))
    
    
class CostFunction:

    def __init__(self, cost_func):
        self.func_type = cost_func
        self.func = self.get_func(cost_func)
    
    def apply(self, expression, factor):
        return self.func(expression, factor)
    
    def __call__(self, expression, factor):
        return self.apply(expression, factor)
    
    def _parabolic(self, expression, factor):
        return factor * np.power(expression, 2) / 2
    
    def _abs(self, expression, factor):
        return factor * abs(expression)
    
    def get_func(self, cost_func):
        if cost_func == "parabolic":
            return self._parabolic
        else:
            return self._abs

class Epistasis:
    
    def __init__(self, epistasis_type):
        self.func_type = epistasis_type
        self.func = self.get_func(epistasis_type)
        
    def __call__(self, *allele_products):
        return self.apply(*allele_products)
    
    def apply(self, *allele_products):
        return self.func(allele_products)
        
    def get_func(self, epistasis_type):
        if epistasis_type == "max":
            return np.max
        elif epistasis_type == "min":
            return np.min
        elif epistasis_type == "sum":
            return np.sum
        else:
            return np.mean