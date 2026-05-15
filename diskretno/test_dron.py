# test_dron.py
from pursuit_evasion import GridWorld, Action
from strategies import EvaderStrategy

def test_movement():
    """Test kretanja drona"""
    print("="*60)
    print("TEST KRETANJA DRONA")
    print("="*60)
    
    env = GridWorld(size=10, obstacles=[])
    
    # Test 1: Direktno kretanje
    print("\n--- TEST 1: Direktno kretanje ---")
    pos = (5, 5)
    print(f"Početak: {pos}")
    
    for action in [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]:
        new_pos = env.apply_evader_action(pos, action)
        print(f"  {action.name}: {pos} -> {new_pos}")
    
    # Test 2: Kretanje sa strategijom
    print("\n--- TEST 2: Kretanje sa strategijom escape_to_goal ---")
    pos = (5, 5)
    pursuer_pos = (0, 0)
    
    for step in range(5):
        action = EvaderStrategy.escape_to_goal(pursuer_pos, pos, env)
        new_pos = env.apply_evader_action(pos, action)
        print(f"  Step {step}: {pos} -> {action.name} -> {new_pos}")
        pos = new_pos
    
    # Test 3: Provjera validnih pozicija
    print("\n--- TEST 3: Provjera granica ---")
    pos = (0, 0)
    print(f"Početak na ivici: {pos}")
    
    # Pokušaj ići van granice
    action = Action.UP
    new_pos = env.apply_evader_action(pos, action)
    print(f"  Pokušaj {action.name} sa {pos}: -> {new_pos} (treba ostati isto)")

if __name__ == "__main__":
    test_movement()