# main.py
import numpy as np
import matplotlib.pyplot as plt
from pursuit_evasion import GridWorld
from strategies import PursuerStrategy, EvaderStrategy, HybridStrategy
from simulation import GameSimulator


def plot_comparison(results_dict):
    """Prikaz poredenja razlicitih strategija"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    strategies = list(results_dict.keys())
    capture_rates = [results_dict[s]['capture_rate'] for s in strategies]
    escape_rates = [results_dict[s]['escape_rate'] for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    ax1.bar(x, capture_rates, width, label='Uhvacen', color='red', alpha=0.7)
    ax1.bar(x, escape_rates, width, bottom=capture_rates, 
            label='Pobegao', color='green', alpha=0.7)
    
    ax1.set_xlabel('Strategije')
    ax1.set_ylabel('Stopa')
    ax1.set_title('Ishod igre - Raketa vs Dron')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    avg_capture = [results_dict[s]['avg_steps_to_capture'] if results_dict[s]['captures'] > 0 else 0 
                   for s in strategies]
    avg_escape = [results_dict[s]['avg_steps_to_escape'] if results_dict[s]['escapes'] > 0 else 0 
                  for s in strategies]
    
    ax2.bar(x, avg_capture, width, label='Prosecno vreme do hvatanja', color='red', alpha=0.7)
    ax2.bar(x, avg_escape, width, label='Prosecno vreme do bega', color='green', alpha=0.7)
    ax2.set_xlabel('Strategije')
    ax2.set_ylabel('Broj koraka')
    ax2.set_title('Prosecno vreme do kraja igre')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('rocket_vs_drone_results.png', dpi=150)
    plt.show()


def run_demo():
    """Pokreni demonstraciju jedne epizode sa GUI kontrolama"""
    print("\n" + "="*60)
    print("DEMONSTRACIJA - Raketa vs Dron")
    print("="*60)
    
    # Pocetno okruzenje sa max 4 prepreke
    obstacles = [(5, 5), (5, 6), (10, 10), (10, 11)]
    env = GridWorld(
        size=15,
        obstacles=obstacles,
        pursuer_speed=3,
        evader_speed=1,
        pursuer_maneuverable=False,
        evader_maneuverable=True,
        capture_radius=1
    )
    
    simulator = GameSimulator(
        env,
        PursuerStrategy.homing_missile,
        EvaderStrategy.escape_to_goal
    )
    
    print("\nKONTROLE U PROZORU:")
    print("   Ponovi igru - resetuje trenutnu epizodu")
    print("   Randomizuj sve - nove pozicije, prepreke i cilj")
    print("   Jedan korak - izvrsava jedan potez")
    print("   Auto igra - automatsko izvrsavanje do kraja")
    print("   Stop - zaustavlja auto igru")
    print("   Zatvori - zatvara prozor")
    print("   Bezbedna zona - ukljuci/iskljuci cilj (zeleno polje)")
    print("="*60)
    
    simulator.render_episode(max_steps=50)


def run_experiments():
    """Pokreni eksperimente"""
    print("\n" + "="*60)
    print("EKSPERIMENTI - Raketa vs Dron")
    print("="*60)
    
    # Maksimalno 4 prepreke
    obstacles = [
        (5, 5), (5, 6), (10, 10), (10, 11)
    ]
    
    env = GridWorld(
        size=20,
        obstacles=obstacles,
        pursuer_speed=3,
        evader_speed=1,
        pursuer_maneuverable=False,
        evader_maneuverable=True,
        capture_radius=1
    )
    
    results = {}
    
    print("\nEKSPERIMENT 1: Homing raketa vs Escape dron")
    sim1 = GameSimulator(env, PursuerStrategy.homing_missile, EvaderStrategy.escape_to_goal)
    results['Homing vs Escape'] = sim1.evaluate(num_episodes=20, max_steps=80)
    print(f"  Stopa hvatanja: {results['Homing vs Escape']['capture_rate']:.2%}")
    print(f"  Stopa bega: {results['Homing vs Escape']['escape_rate']:.2%}")
    
    print("\nEKSPERIMENT 2: Homing raketa vs Max distance dron")
    sim2 = GameSimulator(env, PursuerStrategy.homing_missile, EvaderStrategy.maximize_distance)
    results['Homing vs MaxDist'] = sim2.evaluate(num_episodes=20, max_steps=80)
    print(f"  Stopa hvatanja: {results['Homing vs MaxDist']['capture_rate']:.2%}")
    print(f"  Stopa bega: {results['Homing vs MaxDist']['escape_rate']:.2%}")
    
    print("\nEKSPERIMENT 3: Homing raketa vs Evasive dron")
    sim3 = GameSimulator(env, PursuerStrategy.homing_missile, EvaderStrategy.evasive_maneuver)
    results['Homing vs Evasive'] = sim3.evaluate(num_episodes=20, max_steps=80)
    print(f"  Stopa hvatanja: {results['Homing vs Evasive']['capture_rate']:.2%}")
    print(f"  Stopa bega: {results['Homing vs Evasive']['escape_rate']:.2%}")
    
    plot_comparison(results)
    return results


def main():
    print("\n" + "="*30)
    print("   RAKETA VS DRON - SIMULATOR")
    print("="*30)
    
    while True:
        print("\nIZBORNIK:")
        print("  1. Pokreni demonstraciju (sa GUI kontrolama)")
        print("  2. Pokreni eksperimente (20 epizoda po strategiji)")
        print("  3. Izadji")
        
        choice = input("\nVas izbor (1-3): ").strip()
        
        if choice == '1':
            run_demo()
        elif choice == '2':
            run_experiments()
        elif choice == '3':
            print("\nDovidjenja!")
            break
        else:
            print("\nPogresan izbor")


if __name__ == "__main__":
    main()