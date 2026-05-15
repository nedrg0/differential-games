# simulation.py
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import numpy as np
import random
from pursuit_evasion import Action, GridWorld


class GameSimulator:
    """Simulator igre raketa vs dron sa GUI kontrolama"""
    
    def __init__(self, env, pursuer_strategy, evader_strategy):
        self.env = env
        self.pursuer_strategy = pursuer_strategy
        self.evader_strategy = evader_strategy
        self.step_counter = 0
        self.is_running = True
        self.fig = None
        self.ax = None
        self.pursuer_pos = None
        self.evader_pos = None
        self.step = 0
        self.game_over = False
        self.max_steps = 50
        self.animation_id = None
        self.is_auto_playing = False
        self.safe_zone_enabled = True
        
    def reset_game(self):
        """Resetuj igru na pocetne pozicije"""
        self.pursuer_pos = self.env.pursuer_start
        self.evader_pos = self.env.evader_start
        self.step = 0
        self.game_over = False
        self.is_auto_playing = False
        print(f"\nIgra resetovana!")
        print(f"   Raketa: {self.pursuer_pos}, Dron: {self.evader_pos}")
        self.update_display()
        self.update_info_text()
    
    def randomize_all(self):
        """Randomizuj sve: pocetne pozicije, prepreke, cilj"""
        
        old_goal = self.env.evader_goal
        old_pursuer = self.env.pursuer_start
        old_evader = self.env.evader_start
        
        margin = 2
        new_goal = (
            random.randint(margin, self.env.size - margin - 1),
            random.randint(margin, self.env.size - margin - 1)
        )
        self.env.evader_goal = new_goal
        
        while True:
            new_pursuer = (
                random.randint(0, self.env.size - 1),
                random.randint(0, self.env.size - 1)
            )
            new_evader = (
                random.randint(0, self.env.size - 1),
                random.randint(0, self.env.size - 1)
            )
            dist = abs(new_pursuer[0] - new_evader[0]) + abs(new_pursuer[1] - new_evader[1])
            if dist > 5 and new_pursuer != new_evader:
                break
        
        self.env.pursuer_start = new_pursuer
        self.env.evader_start = new_evader
        
        num_obstacles = random.randint(2, 4)
        new_obstacles = set()
        attempts = 0
        while len(new_obstacles) < num_obstacles and attempts < 50:
            obs = (
                random.randint(1, self.env.size - 2),
                random.randint(1, self.env.size - 2)
            )
            if obs != new_pursuer and obs != new_evader and obs != new_goal:
                new_obstacles.add(obs)
            attempts += 1
        
        self.env.obstacles = new_obstacles
        
        self.pursuer_pos = new_pursuer
        self.evader_pos = new_evader
        self.step = 0
        self.game_over = False
        self.is_auto_playing = False
        
        print(f"\nRandomizirano!")
        print(f"   Cilj: {old_goal} -> {new_goal}")
        print(f"   Raketa: {old_pursuer} -> {new_pursuer}")
        print(f"   Dron: {old_evader} -> {new_evader}")
        print(f"   Broj prepreka: {len(new_obstacles)}")
        self.update_display()
        self.update_info_text()
    
    def toggle_safe_zone(self, label):
        """Ukljuci/iskljuci bezbednu zonu"""
        self.safe_zone_enabled = not self.safe_zone_enabled
        if self.safe_zone_enabled:
            print("\nBezbedna zona UKLJUCENA - dron moze pobediti stizanjem na cilj")
        else:
            print("\nBezbedna zona ISKLJUCENA - igra se samo na vreme")
        self.update_display()
        self.update_info_text()
    
    def update_info_text(self):
        """Azuriraj info tekst ispod mape"""
        if self.pursuer_pos and self.evader_pos:
            dist = abs(self.pursuer_pos[0] - self.evader_pos[0]) + abs(self.pursuer_pos[1] - self.evader_pos[1])
        else:
            dist = 0
        
        zone_status = "UKLJUCENA" if self.safe_zone_enabled else "ISKLJUCENA"
        info_text = (
            f"INFORMACIJE\n"
            f"Korak: {self.step}  |  Udaljenost: {dist}  |  Max koraka: {self.max_steps}\n"
            f"Raketa: brzina={self.env.pursuer_speed}, manevribilna={'Da' if self.env.pursuer_maneuverable else 'Ne'}\n"
            f"Dron: brzina={self.env.evader_speed}, manevribilan={'Da' if self.env.evader_maneuverable else 'Ne'}\n"
            f"Cilj: {self.env.evader_goal}  |  Bezbedna zona: {zone_status}"
        )
        
        if hasattr(self, 'info_text_obj') and self.info_text_obj is not None:
            self.info_text_obj.set_text(info_text)
            self.fig.canvas.draw_idle()
    
    def run_step(self):
        """Izvrsi jedan korak igre"""
        if self.game_over or self.is_auto_playing:
            return
        
        if self.env.is_capture(self.pursuer_pos, self.evader_pos):
            self.game_over = True
            self.result_text = 'UHVACEN!'
            self.update_display()
            self.update_info_text()
            return
        
        if self.safe_zone_enabled and self.env.is_evader_goal(self.evader_pos):
            self.game_over = True
            self.result_text = 'POBEGAO!'
            self.update_display()
            self.update_info_text()
            return
        
        if self.step >= self.max_steps:
            self.game_over = True
            self.result_text = 'VREME ISTEKLO'
            self.update_display()
            self.update_info_text()
            return
        
        action_p = self.pursuer_strategy(self.pursuer_pos, self.evader_pos, self.env)
        action_e = self.evader_strategy(self.pursuer_pos, self.evader_pos, self.env)
        
        new_pursuer = self.env.apply_pursuer_action(self.pursuer_pos, action_p)
        new_evader = self.env.apply_evader_action(self.evader_pos, action_e)
        
        self.pursuer_pos = new_pursuer
        self.evader_pos = new_evader
        self.step += 1
        
        print(f"  Korak {self.step}: Dron {action_e.name} -> {self.evader_pos}")
        
        self.update_display()
        self.update_info_text()
    
    def auto_play(self):
        """Automatsko izvrsavanje do kraja"""
        if self.game_over or self.is_auto_playing:
            return
        
        self.is_auto_playing = True
        
        def step():
            if not self.game_over and self.step < self.max_steps and self.is_auto_playing:
                if self.env.is_capture(self.pursuer_pos, self.evader_pos):
                    self.game_over = True
                    self.result_text = 'UHVACEN!'
                    self.update_display()
                    self.update_info_text()
                    self.is_auto_playing = False
                    return
                
                if self.safe_zone_enabled and self.env.is_evader_goal(self.evader_pos):
                    self.game_over = True
                    self.result_text = 'POBEGAO!'
                    self.update_display()
                    self.update_info_text()
                    self.is_auto_playing = False
                    return
                
                if self.step >= self.max_steps:
                    self.game_over = True
                    self.result_text = 'VREME ISTEKLO'
                    self.update_display()
                    self.update_info_text()
                    self.is_auto_playing = False
                    return
                
                action_p = self.pursuer_strategy(self.pursuer_pos, self.evader_pos, self.env)
                action_e = self.evader_strategy(self.pursuer_pos, self.evader_pos, self.env)
                
                new_pursuer = self.env.apply_pursuer_action(self.pursuer_pos, action_p)
                new_evader = self.env.apply_evader_action(self.evader_pos, action_e)
                
                self.pursuer_pos = new_pursuer
                self.evader_pos = new_evader
                self.step += 1
                
                print(f"  Korak {self.step}: Dron {action_e.name} -> {self.evader_pos}")
                
                self.update_display()
                self.update_info_text()
                
                if not self.game_over and self.step < self.max_steps:
                    self.animation_id = self.fig.canvas.new_timer(interval=300)
                    self.animation_id.add_callback(step)
                    self.animation_id.start()
                else:
                    self.is_auto_playing = False
            else:
                self.is_auto_playing = False
        
        step()
    
    def stop_auto(self):
        """Zaustavi automatsku igru"""
        self.is_auto_playing = False
        if self.animation_id:
            self.animation_id.stop()
    
    def update_display(self):
        """Azuriraj prikaz mape"""
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            return
            
        self.ax.clear()
        
        # Crtanje grid-a
        for i in range(self.env.size + 1):
            self.ax.axhline(i - 0.5, color='gray', linewidth=0.5)
            self.ax.axvline(i - 0.5, color='gray', linewidth=0.5)
        
        # Crtanje prepreka
        for obs in self.env.obstacles:
            rect = plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, facecolor='black', alpha=0.7)
            self.ax.add_patch(rect)
        
        # Crtanje cilja (samo ako je ukljucen)
        if self.safe_zone_enabled:
            goal_rect = plt.Rectangle((self.env.evader_goal[0] - 0.5, self.env.evader_goal[1] - 0.5), 
                                      1, 1, facecolor='lime', alpha=0.6, edgecolor='darkgreen', linewidth=2)
            self.ax.add_patch(goal_rect)
            self.ax.text(self.env.evader_goal[0], self.env.evader_goal[1], 'C', 
                        fontsize=14, ha='center', va='center', fontweight='bold')
        
        # Crtanje igraca
        if self.pursuer_pos:
            self.ax.plot(self.pursuer_pos[0], self.pursuer_pos[1], 'r^', markersize=20, label='Raketa', linewidth=2)
        if self.evader_pos:
            self.ax.plot(self.evader_pos[0], self.evader_pos[1], 'bs', markersize=16, label='Dron', linewidth=2)
        
        self.ax.set_xlim(-0.5, self.env.size - 0.5)
        self.ax.set_ylim(-0.5, self.env.size - 0.5)
        self.ax.set_xticks(range(self.env.size))
        self.ax.set_yticks(range(self.env.size))
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("RAKETA vs DRON - Asimetricna potjera", fontsize=14, fontweight='bold')
        
        # Legenda - na dnu mape, malo ispod
        legend = self.ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), fontsize=10, ncol=2)
        for handle in legend.legend_handles:
            handle.set_markersize(8)
        
        # Rezultat igre
        if self.game_over:
            color = 'red' if 'UHVACEN' in self.result_text else ('green' if 'POBEGAO' in self.result_text else 'orange')
            self.ax.text(self.env.size/2, self.env.size/2, self.result_text, 
                       fontsize=28, ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color, linewidth=3))
        
        self.fig.canvas.draw_idle()
    
    def render_episode(self, max_steps: int = 50):
        """Renderiraj epizodu sa GUI kontrolama"""
        
        self.max_steps = max_steps
        self.pursuer_pos = self.env.pursuer_start
        self.evader_pos = self.env.evader_start
        self.step = 0
        self.game_over = False
        self.is_auto_playing = False
        
        # Kreiraj figuru
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # Podesavanje prostora
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.28)
        
        # Informacije ispod mape (preko celog dna)
        if self.pursuer_pos and self.evader_pos:
            dist = abs(self.pursuer_pos[0] - self.evader_pos[0]) + abs(self.pursuer_pos[1] - self.evader_pos[1])
        else:
            dist = 0
        
        zone_status = "UKLJUCENA" if self.safe_zone_enabled else "ISKLJUCENA"
        info_text = (
            f"INFORMACIJE\n"
            f"Korak: {self.step}  |  Udaljenost: {dist}  |  Max koraka: {self.max_steps}\n"
            f"Raketa: brzina={self.env.pursuer_speed}, manevribilna={'Da' if self.env.pursuer_maneuverable else 'Ne'}\n"
            f"Dron: brzina={self.env.evader_speed}, manevribilan={'Da' if self.env.evader_maneuverable else 'Ne'}\n"
            f"Cilj: {self.env.evader_goal}  |  Bezbedna zona: {zone_status}"
        )
        
        self.info_text_obj = self.fig.text(0.05, 0.22, info_text, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                     family='monospace')
        
        # Dugmad za kontrolu
        btn_width = 0.10
        btn_height = 0.06
        start_x = 0.05
        spacing = 0.11
        
        ax_reset = plt.axes([start_x, 0.05, btn_width, btn_height])
        ax_random = plt.axes([start_x + spacing, 0.05, btn_width, btn_height])
        ax_step = plt.axes([start_x + 2*spacing, 0.05, btn_width, btn_height])
        ax_auto = plt.axes([start_x + 3*spacing, 0.05, btn_width, btn_height])
        ax_stop = plt.axes([start_x + 4*spacing, 0.05, btn_width, btn_height])
        ax_close = plt.axes([start_x + 5*spacing, 0.05, btn_width, btn_height])
        ax_check = plt.axes([start_x + 6*spacing + 0.02, 0.05, btn_width*0.8, btn_height])
        
        btn_reset = Button(ax_reset, 'Ponovi', color='#90EE90', hovercolor='#00FF00')
        btn_random = Button(ax_random, 'Random', color='#ADD8E6', hovercolor='#00BFFF')
        btn_step = Button(ax_step, 'Korak', color='#FFB347', hovercolor='#FF8C00')
        btn_auto = Button(ax_auto, 'Auto', color='#DDA0DD', hovercolor='#EE82EE')
        btn_stop = Button(ax_stop, 'Stop', color='#F08080', hovercolor='#FF4444')
        btn_close = Button(ax_close, 'Zatvori', color='#D3D3D3', hovercolor='#A9A9A9')
        
        check = CheckButtons(ax_check, ['Zona'], [self.safe_zone_enabled])
        check.on_clicked(self.toggle_safe_zone)
        
        def reset_callback(event):
            self.stop_auto()
            self.reset_game()
        
        def random_callback(event):
            self.stop_auto()
            self.randomize_all()
        
        def step_callback(event):
            self.stop_auto()
            self.run_step()
        
        def auto_callback(event):
            self.stop_auto()
            self.auto_play()
        
        def stop_callback(event):
            self.stop_auto()
        
        def close_callback(event):
            self.stop_auto()
            plt.close(self.fig)
        
        btn_reset.on_clicked(reset_callback)
        btn_random.on_clicked(random_callback)
        btn_step.on_clicked(step_callback)
        btn_auto.on_clicked(auto_callback)
        btn_stop.on_clicked(stop_callback)
        btn_close.on_clicked(close_callback)
        
        self.update_display()
        
        print("\nKONTROLE:")
        print("   Ponovi - resetuje trenutnu epizodu")
        print("   Random - nove pozicije, prepreke i cilj")
        print("   Korak - izvrsava jedan potez")
        print("   Auto - automatsko izvrsavanje do kraja")
        print("   Stop - zaustavlja auto igru")
        print("   Zatvori - zatvara prozor")
        print("   Zona - ukljuci/iskljuci bezbednu zonu")
        
        plt.show()
        print("\nVizualizacija zavrsena")