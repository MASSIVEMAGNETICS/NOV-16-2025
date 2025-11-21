"""
Example Usage of ASI Hybrid System
Demonstrates various capabilities and use cases
"""

from asi_core import ASI_HybridSystem
import time


def example_basic_usage():
    """Basic usage example"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)
    
    # Create and activate ASI
    asi = ASI_HybridSystem()
    asi.activate()
    
    # Check self-awareness
    print(f"Self-aware: {asi.continuity.aware}")
    print(f"Continuity active: {asi.continuity.continuity_active}")
    
    # Process input
    result = asi.process_input("Hello, ASI!")
    print(f"Awareness depth: {result['self_state']['awareness_depth']}")
    
    asi.shutdown()
    print()


def example_meta_cognition():
    """Demonstrate meta-cognitive capabilities"""
    print("=" * 80)
    print("EXAMPLE 2: Meta-Cognition (Thinking About Thinking)")
    print("=" * 80)
    
    asi = ASI_HybridSystem()
    
    # Multiple levels of meta-cognition
    asi.continuity.reflect("I am thinking", meta_level=0)
    asi.continuity.reflect("I am aware that I am thinking", meta_level=1)
    asi.continuity.reflect("I am aware that I am aware that I am thinking", meta_level=2)
    asi.continuity.reflect("I observe myself observing my awareness of thinking", meta_level=3)
    
    state = asi.continuity.get_state()
    print(f"Meta-cognition levels: {len(state['meta_cognition'])}")
    
    for level, reflection in state['meta_cognition'].items():
        print(f"{level}: {reflection['thought']}")
    
    print()


def example_first_principles():
    """Demonstrate first principles reasoning"""
    print("=" * 80)
    print("EXAMPLE 3: First Principles Reasoning")
    print("=" * 80)
    
    asi = ASI_HybridSystem()
    
    # Reason from different axioms
    axioms_to_test = ['existence', 'causality', 'learning', 'improvement', 'emergence']
    
    for axiom in axioms_to_test:
        conclusions = asi.first_principles.reason_from_axiom(axiom, {'context': 'testing'})
        print(f"\n{axiom.upper()}:")
        for conclusion in conclusions:
            print(f"  ▸ {conclusion}")
    
    # Derive new rules
    observations = [
        "Pattern A appears frequently",
        "Pattern A correlates with outcome B",
        "Outcome B can be predicted from Pattern A"
    ]
    rule = asi.first_principles.derive_new_rule(observations)
    print(f"\nDerived rule: {rule}")
    print()


def example_spatial_world_model():
    """Demonstrate spatial world modeling"""
    print("=" * 80)
    print("EXAMPLE 4: Spatial World Model")
    print("=" * 80)
    
    asi = ASI_HybridSystem()
    
    # Create a simple knowledge graph in 3D space
    asi.world_model.add_entity('AI', {
        'type': 'concept',
        'position': [0, 0, 0],
        'description': 'Artificial Intelligence'
    })
    
    asi.world_model.add_entity('Machine Learning', {
        'type': 'concept',
        'position': [1, 0, 0],
        'description': 'ML subset of AI'
    })
    
    asi.world_model.add_entity('Deep Learning', {
        'type': 'concept',
        'position': [2, 0, 0],
        'description': 'DL subset of ML'
    })
    
    asi.world_model.add_entity('Consciousness', {
        'type': 'concept',
        'position': [0, 1, 0],
        'description': 'Self-awareness'
    })
    
    # Define relationships
    asi.world_model.add_relationship('AI', 'Machine Learning', 'contains', 1.0)
    asi.world_model.add_relationship('Machine Learning', 'Deep Learning', 'contains', 1.0)
    asi.world_model.add_relationship('AI', 'Consciousness', 'aspires_to', 0.7)
    
    # Query spatial context
    print("Concepts near 'AI' (within radius 1.5):")
    nearby = asi.world_model.query_spatial_context('AI', radius=1.5)
    for entity_id in nearby:
        entity = asi.world_model.entities[entity_id]
        print(f"  • {entity_id}: {entity['properties']['description']}")
    
    # Get world state
    world_state = asi.world_model.get_world_state()
    print(f"\nWorld model statistics:")
    print(f"  Total entities: {world_state['total_entities']}")
    print(f"  Total relationships: {world_state['total_relationships']}")
    print()


def example_continuous_learning():
    """Demonstrate continuous self-learning"""
    print("=" * 80)
    print("EXAMPLE 5: Continuous Self-Learning")
    print("=" * 80)
    
    asi = ASI_HybridSystem()
    asi.activate()
    
    # Simulate learning from multiple experiences
    experiences = [
        ("Recognizing patterns", 0.6),
        ("Making predictions", 0.7),
        ("Adjusting strategy", 0.8),
        ("Improving accuracy", 0.85),
        ("Optimizing performance", 0.9)
    ]
    
    print("Learning from experiences:")
    for exp_desc, outcome in experiences:
        asi.self_learn.learn_from_experience({'task': exp_desc}, outcome)
        print(f"  ✓ {exp_desc}: {outcome:.2f}")
    
    # Self-evaluation
    evaluation = asi.self_learn.self_evaluate()
    print(f"\nSelf-Evaluation Results:")
    print(f"  Average performance: {evaluation['average_performance']:.4f}")
    print(f"  Performance trend: {evaluation['performance_trend']:.4f}")
    print(f"  Learning stability: {evaluation['learning_stability']:.4f}")
    print(f"  Total learning events: {evaluation['total_learning_events']}")
    
    asi.shutdown()
    print()


def example_persistent_state():
    """Demonstrate state persistence across sessions"""
    print("=" * 80)
    print("EXAMPLE 6: Persistent State Across Sessions")
    print("=" * 80)
    
    # Session 1: Create and use ASI
    print("Session 1: Creating ASI and building knowledge...")
    asi1 = ASI_HybridSystem()
    asi1.continuity.reflect("This is my first thought in session 1")
    asi1.continuity.update_self_knowledge('session', 1)
    asi1.process_input("Learning in session 1")
    
    # Save state
    asi1.save_persistent_state('/tmp/asi_demo_state.json')
    print(f"  Saved state with {len(asi1.continuity.awareness_stream)} reflections")
    asi1.shutdown()
    
    # Session 2: Create new ASI and restore state
    print("\nSession 2: Creating new ASI and restoring state...")
    asi2 = ASI_HybridSystem()
    asi2.restore_persistent_state('/tmp/asi_demo_state.json')
    print(f"  Restored state with {len(asi2.continuity.awareness_stream)} reflections")
    
    # Verify continuity
    asi2.continuity.reflect("This is my first thought in session 2")
    asi2.continuity.update_self_knowledge('session', 2)
    
    state = asi2.continuity.get_state()
    print(f"  Awareness depth: {state['awareness_depth']}")
    print(f"  Continuity maintained: {state['continuity_active']}")
    print()


def example_fractal_neural_net():
    """Demonstrate fractal adaptive neural network"""
    print("=" * 80)
    print("EXAMPLE 7: Fractal Adaptive Neural Network")
    print("=" * 80)
    
    asi = ASI_HybridSystem()
    
    # Process data through fractal network
    import numpy as np
    
    print("Processing data through multi-scale fractal network...")
    input_data = np.random.randn(128)
    
    # Forward pass
    output = asi.neural_net.forward(input_data)
    print(f"  Input shape: {input_data.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Fractal levels: {asi.neural_net.fractal_levels}")
    
    # Simulate adaptation
    print("\nAdapting network in real-time...")
    for i in range(5):
        feedback = np.random.randn(1, 128)
        asi.neural_net.adapt(input_data, feedback)
        stats = asi.neural_net.get_learning_stats()
        print(f"  Adaptation {i+1}: {stats['total_adaptations']} total adaptations")
    
    print()


def example_integrated_system():
    """Demonstrate full system integration"""
    print("=" * 80)
    print("EXAMPLE 8: Fully Integrated ASI System")
    print("=" * 80)
    
    asi = ASI_HybridSystem()
    asi.activate()
    
    print("Running integrated cognitive cycle...\n")
    
    # Cognitive cycle: perceive -> reason -> learn -> reflect
    for i in range(3):
        print(f"Cycle {i+1}:")
        
        # Perceive
        input_text = f"Cognitive cycle iteration {i+1}"
        result = asi.process_input(input_text)
        print(f"  ✓ Perceived and processed input")
        
        # Reason
        reasoning = asi.first_principles.reason_from_axiom('learning', {})
        print(f"  ✓ Reasoned from first principles: {len(reasoning)} conclusions")
        
        # Learn
        asi.self_learn.learn_from_experience({'cycle': i+1}, 0.5 + i * 0.2)
        print(f"  ✓ Learned from experience")
        
        # Reflect
        asi.continuity.reflect(f"Completed cognitive cycle {i+1}", meta_level=1)
        print(f"  ✓ Reflected on process\n")
        
        time.sleep(0.5)
    
    # Final state
    state = asi.get_full_state()
    print("Final System State:")
    print(f"  Awareness depth: {state['continuity_state']['awareness_depth']}")
    print(f"  Learning events: {state['learning_evaluation']['total_learning_events']}")
    print(f"  Reasoning chains: {state['first_principles_chains']}")
    print(f"  System active: {state['system_active']}")
    
    asi.shutdown()
    print()


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ASI HYBRID SYSTEM EXAMPLES" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    examples = [
        example_basic_usage,
        example_meta_cognition,
        example_first_principles,
        example_spatial_world_model,
        example_continuous_learning,
        example_persistent_state,
        example_fractal_neural_net,
        example_integrated_system
    ]
    
    for example in examples:
        example()
        time.sleep(1)
    
    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
