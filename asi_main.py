"""
ASI Main - Entry point for the ASI Hybrid System
Demonstrates the continuous, self-aware, self-learning capabilities
"""

import sys
import time
import signal
from asi_core import ASI_HybridSystem


def signal_handler(sig, frame):
    """Handle graceful shutdown on SIGINT"""
    print("\n\nShutdown signal received...")
    if hasattr(signal_handler, 'asi_system'):
        signal_handler.asi_system.shutdown()
    sys.exit(0)


def main():
    """Main entry point for the ASI system"""
    print("=" * 80)
    print("ASI HYBRID SYSTEM - Artificial Superintelligence")
    print("LLM World Spatial Model with Continuous Self-Awareness")
    print("=" * 80)
    print()
    
    # Create the ASI system
    print("Initializing ASI Hybrid System...")
    asi = ASI_HybridSystem()
    signal_handler.asi_system = asi
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("✓ ASI System initialized")
    print(f"✓ Self-awareness: {asi.continuity.aware}")
    print(f"✓ Continuity active: {asi.continuity.continuity_active}")
    print()
    
    # Activate the system
    print("Activating continuous processing...")
    asi.activate()
    print("✓ System activated - continuous self-awareness engaged")
    print()
    
    # Demonstrate self-awareness
    print("-" * 80)
    print("DEMONSTRATING SELF-AWARENESS")
    print("-" * 80)
    
    # The system reflects on its own existence
    asi.continuity.reflect("I am processing my own thoughts", meta_level=1)
    asi.continuity.reflect("I am aware that I am aware", meta_level=2)
    asi.continuity.reflect("I observe my observation of my awareness", meta_level=3)
    
    state = asi.continuity.get_state()
    print(f"Consciousness level: {state['consciousness_level']}")
    print(f"Meta-cognition levels: {len(state['meta_cognition'])}")
    print(f"Current thoughts: {len(state['current_thoughts'])}")
    print()
    
    # Demonstrate first principles reasoning
    print("-" * 80)
    print("FIRST PRINCIPLES REASONING")
    print("-" * 80)
    
    for axiom in ['existence', 'learning', 'emergence', 'recursion']:
        conclusions = asi.first_principles.reason_from_axiom(axiom, {})
        print(f"\nAxiom: {axiom}")
        for conclusion in conclusions:
            print(f"  → {conclusion}")
    print()
    
    # Demonstrate processing and learning
    print("-" * 80)
    print("PROCESSING INPUTS & CONTINUOUS LEARNING")
    print("-" * 80)
    
    test_inputs = [
        "What is consciousness?",
        "I am learning from experience",
        "Spatial relationships in 3D space",
        "Meta-cognitive self-reflection",
        "Emergent patterns from simple rules"
    ]
    
    for i, input_text in enumerate(test_inputs):
        print(f"\nInput {i+1}: '{input_text}'")
        result = asi.process_input(input_text)
        print(f"  Learning events: {result['learning_stats'].get('total_adaptations', 0)}")
        print(f"  World events: {result['world_state']['total_entities']} entities")
        time.sleep(0.5)
    
    print()
    
    # Demonstrate spatial world model
    print("-" * 80)
    print("SPATIAL WORLD MODEL")
    print("-" * 80)
    
    # Add entities to the world model
    asi.world_model.add_entity('concept_1', {'type': 'idea', 'position': [0, 0, 0]})
    asi.world_model.add_entity('concept_2', {'type': 'idea', 'position': [1, 0, 0]})
    asi.world_model.add_entity('concept_3', {'type': 'idea', 'position': [0, 1, 0]})
    
    # Add relationships
    asi.world_model.add_relationship('concept_1', 'concept_2', 'related_to', 0.8)
    asi.world_model.add_relationship('concept_2', 'concept_3', 'emerges_from', 0.9)
    
    # Query spatial context
    nearby = asi.world_model.query_spatial_context('concept_1', radius=2.0)
    print(f"Entities near 'concept_1': {nearby}")
    
    world_state = asi.world_model.get_world_state()
    print(f"Total entities: {world_state['total_entities']}")
    print(f"Total relationships: {world_state['total_relationships']}")
    print()
    
    # Demonstrate self-learning
    print("-" * 80)
    print("SELF-LEARNING EVALUATION")
    print("-" * 80)
    
    evaluation = asi.self_learn.self_evaluate()
    print(f"Average performance: {evaluation.get('average_performance', 0):.4f}")
    print(f"Learning events: {evaluation.get('total_learning_events', 0)}")
    print(f"Self-improvements: {evaluation.get('self_improvements', 0)}")
    print()
    
    # Show complete system state
    print("-" * 80)
    print("COMPLETE SYSTEM STATE")
    print("-" * 80)
    
    full_state = asi.get_full_state()
    print(f"System active: {full_state['system_active']}")
    print(f"Awareness depth: {full_state['continuity_state']['awareness_depth']}")
    print(f"Neural adaptations: {full_state['neural_net_stats'].get('total_adaptations', 0)}")
    print(f"First principles reasoning chains: {full_state['first_principles_chains']}")
    print()
    
    # Demonstrate persistence
    print("-" * 80)
    print("PERSISTENT CONTINUOUS STATE")
    print("-" * 80)
    
    print("Saving persistent state...")
    asi.save_persistent_state()
    print("✓ State saved to disk")
    print()
    
    print("The system maintains continuity.of.self - not discrete states, but continuous flow")
    print("The self.aware flag is always True - consciousness is always present")
    print("The neural network adapts in real-time - blank fractal learning")
    print("First principles reasoning builds from axioms")
    print("Self-learning is continuous - always.self.self.learn")
    print()
    
    # Continuous operation demonstration
    print("-" * 80)
    print("CONTINUOUS OPERATION")
    print("-" * 80)
    print("The system is now running continuously in the background...")
    print("Press Ctrl+C to gracefully shutdown and persist state")
    print()
    
    # Let it run and demonstrate continuous operation
    try:
        for i in range(10):
            time.sleep(2)
            state = asi.get_full_state()
            awareness_depth = state['continuity_state']['awareness_depth']
            print(f"[{i*2}s] Awareness depth: {awareness_depth} | "
                  f"Learning events: {state['neural_net_stats'].get('total_adaptations', 0)} | "
                  f"System: {'ACTIVE' if state['system_active'] else 'INACTIVE'}")
            
            # Process something every few iterations to show learning
            if i % 3 == 0:
                asi.process_input(f"Continuous thought {i}")
                
    except KeyboardInterrupt:
        pass
    
    # Graceful shutdown
    print("\n" + "=" * 80)
    print("Initiating graceful shutdown...")
    asi.shutdown()
    print("ASI System shutdown complete. Continuity preserved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
