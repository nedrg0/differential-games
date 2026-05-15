# value_iteration.py
import numpy as np
from typing import Dict, Tuple
from collections import defaultdict
from pursuit_evasion import Action

class ValueIterationSolver:
    """Rješavanje Markov igre pomoću value iteration"""
    
    def __init__(self, env, gamma: float = 0.95, theta: float = 1e-4):
        """
        Args:
            env: GridWorld okruženje
            gamma: Discount faktor
            theta: Tolerancija za konvergenciju
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        # Vrijednosna funkcija
        self.V = defaultdict(float)
        
        # Optimalne politike
        self.pursuer_policy = {}
        self.evader_policy = {}
        
    def get_all_states(self):
        """Generiši sva moguća stanja"""
        states = []
        for px in range(self.env.size):
            for py in range(self.env.size):
                for ex in range(self.env.size):
                    for ey in range(self.env.size):
                        pos_p = (px, py)
                        pos_e = (ex, ey)
                        if (self.env.is_valid_position(pos_p) and 
                            self.env.is_valid_position(pos_e) and
                            not self.env.is_capture(pos_p, pos_e)):
                            states.append((pos_p, pos_e))
        return states
    
    def get_reward(self, pos_p: Tuple[int, int], pos_e: Tuple[int, int]) -> float:
        """Izračunaj nagradu za dato stanje"""
        if self.env.is_capture(pos_p, pos_e):
            return 1.0  # Pursuer dobija +1
        elif self.env.is_evader_goal(pos_e):
            return -1.0  # Evader dobija -1 (pursuer gubi)
        return 0.0
    
    def value_iteration(self, max_iterations: int = 1000):
        """Value iteration algoritam za zero-sum Markov igru"""
        
        states = self.get_all_states()
        
        for iteration in range(max_iterations):
            delta = 0
            
            for pos_p, pos_e in states:
                v_old = self.V[(pos_p, pos_e)]
                
                # Pursuer bira akciju (max), evader bira akciju (min)
                best_value = -float('inf')
                
                for action_p in Action.get_actions():
                    new_pos_p = self.env.apply_action(pos_p, action_p)
                    
                    # Evader minimizira (jer je zero-sum)
                    min_value = float('inf')
                    
                    for action_e in Action.get_actions():
                        new_pos_e = self.env.apply_action(pos_e, action_e)
                        
                        # Izračunaj nagradu i sljedeću vrijednost
                        reward = self.get_reward(new_pos_p, new_pos_e)
                        
                        if self.env.is_capture(new_pos_p, new_pos_e):
                            next_value = reward
                        elif self.env.is_evader_goal(new_pos_e):
                            next_value = reward
                        else:
                            next_value = reward + self.gamma * self.V[(new_pos_p, new_pos_e)]
                        
                        min_value = min(min_value, next_value)
                    
                    best_value = max(best_value, min_value)
                
                self.V[(pos_p, pos_e)] = best_value
                delta = max(delta, abs(v_old - self.V[(pos_p, pos_e)]))
            
            if delta < self.theta:
                print(f"Konvergirao nakon {iteration + 1} iteracija")
                break
        
        # Ekstraktuj politike
        self.extract_policies(states)
    
    def extract_policies(self, states):
        """Ekstrakcija optimalnih politika iz vrijednosne funkcije"""
        
        for pos_p, pos_e in states:
            # Optimalna politika za pursuer
            best_action_p = None
            best_value = -float('inf')
            
            for action_p in Action.get_actions():
                new_pos_p = self.env.apply_action(pos_p, action_p)
                min_value = float('inf')
                
                for action_e in Action.get_actions():
                    new_pos_e = self.env.apply_action(pos_e, action_e)
                    reward = self.get_reward(new_pos_p, new_pos_e)
                    
                    if self.env.is_capture(new_pos_p, new_pos_e):
                        next_value = reward
                    elif self.env.is_evader_goal(new_pos_e):
                        next_value = reward
                    else:
                        next_value = reward + self.gamma * self.V[(new_pos_p, new_pos_e)]
                    
                    min_value = min(min_value, next_value)
                
                if min_value > best_value:
                    best_value = min_value
                    best_action_p = action_p
            
            self.pursuer_policy[(pos_p, pos_e)] = best_action_p
            
            # Optimalna politika za evader
            best_action_e = None
            best_value_e = float('inf')
            
            for action_e in Action.get_actions():
                new_pos_e = self.env.apply_action(pos_e, action_e)
                max_value = -float('inf')
                
                for action_p in Action.get_actions():
                    new_pos_p = self.env.apply_action(pos_p, action_p)
                    reward = self.get_reward(new_pos_p, new_pos_e)
                    
                    if self.env.is_capture(new_pos_p, new_pos_e):
                        next_value = reward
                    elif self.env.is_evader_goal(new_pos_e):
                        next_value = reward
                    else:
                        next_value = reward + self.gamma * self.V[(new_pos_p, new_pos_e)]
                    
                    max_value = max(max_value, next_value)
                
                if max_value < best_value_e:
                    best_value_e = max_value
                    best_action_e = action_e
            
            self.evader_policy[(pos_p, pos_e)] = best_action_e