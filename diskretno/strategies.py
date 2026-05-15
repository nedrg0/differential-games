# strategies.py
from typing import Tuple, List
import random
import math
from pursuit_evasion import Action


class PursuerStrategy:
    """Strategije za raketu (brza, ali manje manevribilna)"""
    
    @staticmethod
    def random_move(pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int], 
                    env) -> Action:
        """Nasumično kretanje"""
        return random.choice(Action.get_basic_actions())
    
    @staticmethod
    def greedy(pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int], 
               env) -> Action:
        """Pohlepna strategija - smanji Manhattan udaljenost"""
        if env.is_capture(pursuer_pos, evader_pos):
            return Action.STAY
        
        best_action = Action.STAY
        best_dist = float('inf')
        
        for action in Action.get_basic_actions():
            new_pos = env.apply_pursuer_action(pursuer_pos, action)
            dist = abs(new_pos[0] - evader_pos[0]) + abs(new_pos[1] - evader_pos[1])
            if dist < best_dist:
                best_dist = dist
                best_action = action
        
        return best_action
    
    @staticmethod
    def homing_missile(pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int],
                       env) -> Action:
        """Strategija navođenja - direktno prema evaderu"""
        if env.is_capture(pursuer_pos, evader_pos):
            return Action.STAY
        
        dx = evader_pos[0] - pursuer_pos[0]
        dy = evader_pos[1] - pursuer_pos[1]
        
        if dx == 0 and dy == 0:
            return Action.STAY
        
        if abs(dx) > abs(dy):
            return Action.RIGHT if dx > 0 else Action.LEFT
        else:
            return Action.UP if dy > 0 else Action.DOWN
    
    @staticmethod
    def predictive_intercept(pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int],
                             env) -> Action:
        """Prediktivno presretanje"""
        if env.is_capture(pursuer_pos, evader_pos):
            return Action.STAY
        
        best_action = Action.STAY
        min_dist = float('inf')
        
        for action in Action.get_basic_actions():
            new_pursuer = env.apply_pursuer_action(pursuer_pos, action)
            
            # Predvidi gdje će evader biti
            future_evader = PursuerStrategy._predict_evader(evader_pos, env)
            
            dist = abs(new_pursuer[0] - future_evader[0]) + abs(new_pursuer[1] - future_evader[1])
            if dist < min_dist:
                min_dist = dist
                best_action = action
        
        return best_action
    
    @staticmethod
    def _predict_evader(evader_pos: Tuple[int, int], env, steps: int = 2) -> Tuple[int, int]:
        """Predvidi kretanje evadera prema cilju"""
        pos = evader_pos
        for _ in range(steps):
            goal = env.evader_goal
            dx = goal[0] - pos[0]
            dy = goal[1] - pos[1]
            
            if abs(dx) > abs(dy):
                pos = (pos[0] + (1 if dx > 0 else -1), pos[1])
            else:
                pos = (pos[0], pos[1] + (1 if dy > 0 else -1))
        return pos


class EvaderStrategy:
    """Strategije za dron (sporiji, ali potpuno manevribilan)"""
    
    @staticmethod
    def random_move(pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int],
                    env) -> Action:
        """Nasumično kretanje"""
        if env.is_capture(pursuer_pos, evader_pos):
            return Action.STAY
        if env.is_evader_goal(evader_pos):
            return Action.STAY
        return random.choice(Action.get_all_actions())
    
    @staticmethod
    def escape_to_goal(pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int],
                       env) -> Action:
        """Bježi prema cilju"""
        if env.is_capture(pursuer_pos, evader_pos):
            return Action.STAY
        if env.is_evader_goal(evader_pos):
            return Action.STAY
        
        goal = env.evader_goal
        dx = goal[0] - evader_pos[0]
        dy = goal[1] - evader_pos[1]
        
        if dx == 0 and dy == 0:
            return Action.STAY
        
        if abs(dx) > abs(dy):
            return Action.RIGHT if dx > 0 else Action.LEFT
        else:
            return Action.UP if dy > 0 else Action.DOWN
    
    @staticmethod
    def maximize_distance(pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int],
                          env) -> Action:
        """Maksimiziraj udaljenost od potjere"""
        if env.is_capture(pursuer_pos, evader_pos):
            return Action.STAY
        
        best_action = Action.STAY
        best_dist = -float('inf')
        
        for action in Action.get_all_actions():
            if action == Action.STAY:
                new_pos = evader_pos
            else:
                dx, dy = action.value
                new_pos = (evader_pos[0] + dx, evader_pos[1] + dy)
            
            if not env.is_valid_position(new_pos):
                continue
            
            dist = abs(new_pos[0] - pursuer_pos[0]) + abs(new_pos[1] - pursuer_pos[1])
            if dist > best_dist:
                best_dist = dist
                best_action = action
        
        return best_action
    
    @staticmethod
    def evasive_maneuver(pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int],
                         env) -> Action:
        """Evasive manevar - bježi okomito na pravac rakete"""
        if env.is_capture(pursuer_pos, evader_pos):
            return Action.STAY
        
        dx = evader_pos[0] - pursuer_pos[0]
        dy = evader_pos[1] - pursuer_pos[1]
        
        if dx == 0 and dy == 0:
            return Action.STAY
        
        if abs(dx) > abs(dy):
            return Action.UP if dy >= 0 else Action.DOWN
        else:
            return Action.RIGHT if dx >= 0 else Action.LEFT
    
    @staticmethod
    def optimize_separation(pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int],
                           env) -> Action:
        """Optimizuj separaciju"""
        return EvaderStrategy.maximize_distance(pursuer_pos, evader_pos, env)


class HybridStrategy:
    """Kombinovane strategije"""
    
    @staticmethod
    def adaptive_pursuer(pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int],
                         env) -> Action:
        """Adaptivna raketa"""
        dist = abs(pursuer_pos[0] - evader_pos[0]) + abs(pursuer_pos[1] - evader_pos[1])
        
        if dist <= env.pursuer_speed * 3:
            return PursuerStrategy.homing_missile(pursuer_pos, evader_pos, env)
        else:
            return PursuerStrategy.predictive_intercept(pursuer_pos, evader_pos, env)
    
    @staticmethod
    def adaptive_evader(pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int],
                        env) -> Action:
        """Adaptivni dron"""
        if env.is_capture(pursuer_pos, evader_pos):
            return Action.STAY
        if env.is_evader_goal(evader_pos):
            return Action.STAY
        
        dist_to_pursuer = abs(evader_pos[0] - pursuer_pos[0]) + abs(evader_pos[1] - pursuer_pos[1])
        dist_to_goal = abs(env.evader_goal[0] - evader_pos[0]) + abs(env.evader_goal[1] - evader_pos[1])
        
        if dist_to_pursuer <= env.pursuer_speed * 2:
            return EvaderStrategy.evasive_maneuver(pursuer_pos, evader_pos, env)
        elif dist_to_goal <= 5:
            return EvaderStrategy.escape_to_goal(pursuer_pos, evader_pos, env)
        else:
            return EvaderStrategy.maximize_distance(pursuer_pos, evader_pos, env)