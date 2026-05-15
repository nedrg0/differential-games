# pursuit_evasion.py
import numpy as np
from enum import Enum
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

class Action(Enum):
    """Moguće akcije za igrače"""
    UP = (0, 1)
    DOWN = (0, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP_LEFT = (-1, 1)
    UP_RIGHT = (1, 1)
    DOWN_LEFT = (-1, -1)
    DOWN_RIGHT = (1, -1)
    STAY = (0, 0)
    
    @staticmethod
    def get_basic_actions():
        """Osnovne akcije (ne dijagonalne) - za manje manevribilne igrače"""
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY]
    
    @staticmethod
    def get_all_actions():
        """Sve akcije uključujući dijagonale - za potpuno manevribilne igrače"""
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT,
                Action.UP_LEFT, Action.UP_RIGHT, Action.DOWN_LEFT, Action.DOWN_RIGHT,
                Action.STAY]
    
    @staticmethod
    def to_vector(action):
        return action.value


class GridWorld:
    """2D grid okruženje za igru potjere i bijega sa asimetričnim kretanjem"""
    
    def __init__(self, size: int = 15, obstacles: List[Tuple[int, int]] = None,
                 pursuer_speed: int = 3, evader_speed: int = 1,
                 pursuer_maneuverable: bool = False, evader_maneuverable: bool = True,
                 capture_radius: int = 1):
        """
        Args:
            size: Veličina grid-a (size x size)
            obstacles: Lista pozicija prepreka
            pursuer_speed: Brzina potjere (broj polja po potezu)
            evader_speed: Brzina bjegunca (broj polja po potezu)
            pursuer_maneuverable: Da li potjera može ići dijagonalno
            evader_maneuverable: Da li bjegunac može ići dijagonalno
            capture_radius: Radijus hvatanja (Manhattan distance)
        """
        self.size = size
        self.obstacles = set(obstacles) if obstacles else set()
        
        # Asimetrične sposobnosti
        self.pursuer_speed = pursuer_speed
        self.evader_speed = evader_speed
        self.pursuer_maneuverable = pursuer_maneuverable
        self.evader_maneuverable = evader_maneuverable
        self.capture_radius = capture_radius
        
        # Cilj za evadera (random)
        self.evader_goal = self._generate_random_goal()
        
        # Početne pozicije
        self.pursuer_start = (0, 0)
        self.evader_start = (size-1, size-1)
        
        # Zadnja akcija za potjeru (za inerciju)
        self.last_pursuer_action = Action.STAY
        
        print(f"Okruženje kreirano: {size}x{size}")
        print(f"Pursuer: brzina={pursuer_speed}, manevribilan={pursuer_maneuverable}")
        print(f"Evader: brzina={evader_speed}, manevribilan={evader_maneuverable}")
        print(f"Cilj evadera: {self.evader_goal}")
        print(f"Početna pozicija drona: {self.evader_start}")
    
    def _generate_random_goal(self) -> Tuple[int, int]:
        """Generiši random cilj za evadera"""
        margin = 2
        x = random.randint(margin, self.size - margin - 1)
        y = random.randint(margin, self.size - margin - 1)
        return (x, y)
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Provjera da li je pozicija validna"""
        x, y = pos
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        if pos in self.obstacles:
            return False
        return True
    
    def is_capture(self, pursuer_pos: Tuple[int, int], evader_pos: Tuple[int, int]) -> bool:
        """Provjera da li je došlo do hvatanja"""
        manhattan_dist = abs(pursuer_pos[0] - evader_pos[0]) + abs(pursuer_pos[1] - evader_pos[1])
        return manhattan_dist <= self.capture_radius
    
    def is_evader_goal(self, evader_pos: Tuple[int, int]) -> bool:
        """Provjera da li je evader stigao do cilja"""
        return evader_pos == self.evader_goal
    
    def apply_action_with_speed(self, pos: Tuple[int, int], action: Action, 
                                 speed: int, maneuverable: bool) -> Tuple[int, int]:
        """
        Primijeni akciju sa brzinom i manevribilnošću.
        
        Args:
            pos: Trenutna pozicija (x, y)
            action: Akcija koja se primjenjuje
            speed: Brzina kretanja (broj polja po potezu)
            maneuverable: Da li igrač može ići dijagonalno
        
        Returns:
            Nova pozicija nakon primjene akcije
        """
        # Ako je akcija STAY, ostani na mjestu
        if action == Action.STAY:
            return pos
        
        # Dobij vektor kretanja
        dx, dy = action.value
        
        # AKO NIJE MANEVRIBILAN (ne može dijagonalno)
        if not maneuverable:
            # Ako je pokušao dijagonalnu akciju, konvertuj je u osnovnu
            if action not in Action.get_basic_actions():
                if abs(dx) > 0 and abs(dy) > 0:  # Dijagonalno
                    # Zadrži dominantni pravac
                    if abs(dx) > abs(dy):
                        action = Action.RIGHT if dx > 0 else Action.LEFT
                    else:
                        action = Action.UP if dy > 0 else Action.DOWN
                    dx, dy = action.value
        
        # Primijeni brzinu KORAK PO KORAK
        current_pos = pos
        
        for step in range(speed):
            # Izračunaj sljedeću poziciju (pomjeri se za 1 polje)
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Provjeri da li je sljedeća pozicija validna
            if not self.is_valid_position(next_pos):
                # Ako nije validna, zaustavi se na trenutnoj poziciji
                break
            
            # Ako jeste validna, pomjeri se
            current_pos = next_pos
        
        return current_pos
    
    def apply_pursuer_action(self, pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
        """Primijeni akciju za potjeru"""
        new_pos = self.apply_action_with_speed(pos, action, self.pursuer_speed, self.pursuer_maneuverable)
        self.last_pursuer_action = action
        return new_pos
    
    def apply_evader_action(self, pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
        """Primijeni akciju za bjegunca"""
        return self.apply_action_with_speed(pos, action, self.evader_speed, self.evader_maneuverable)
    
    def set_obstacles(self, obstacles):
        """Postavi nove prepreke"""
        self.obstacles = set(obstacles)
    
    def set_evader_goal(self, goal):
        """Postavi novi cilj za evadera"""
        self.evader_goal = goal
    
    def set_pursuer_start(self, pos):
        """Postavi novu početnu poziciju za raketu"""
        self.pursuer_start = pos
    
    def set_evader_start(self, pos):
        """Postavi novu početnu poziciju za dron"""
        self.evader_start = pos