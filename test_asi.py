"""
Test Suite for ASI Hybrid System
Tests all core components and their integration
"""

import unittest
import numpy as np
import time
import os
import tempfile
from asi_core import (
    ContinuityOfSelf,
    FirstPrinciplesReasoning,
    FractalAdaptiveNeuralNet,
    SpatialWorldModel,
    AlwaysSelfLearn,
    ASI_HybridSystem
)


class TestContinuityOfSelf(unittest.TestCase):
    """Test the continuous self-awareness component"""
    
    def setUp(self):
        self.continuity = ContinuityOfSelf()
        
    def test_always_aware(self):
        """Test that self.aware is always True"""
        self.assertTrue(self.continuity.aware)
        
    def test_persistent_flag(self):
        """Test that persistent flag is set"""
        self.assertTrue(self.continuity.persistent)
        
    def test_reflection(self):
        """Test self-reflection mechanism"""
        self.continuity.reflect("Test thought", meta_level=0)
        self.assertEqual(len(self.continuity.awareness_stream), 1)
        
        # Test meta-cognition
        self.continuity.reflect("Thinking about thinking", meta_level=1)
        self.assertIn('level_1', self.continuity.self_state['meta_cognition'])
        
    def test_self_knowledge_update(self):
        """Test updating self-knowledge"""
        self.continuity.update_self_knowledge('test_key', 'test_value')
        self.assertIn('test_key', self.continuity.self_state['self_knowledge'])
        
    def test_state_retrieval(self):
        """Test getting current state"""
        state = self.continuity.get_state()
        self.assertIn('identity', state)
        self.assertIn('awareness_depth', state)
        self.assertTrue(state['is_aware'])
        
    def test_state_persistence(self):
        """Test state save/restore"""
        # Add some state
        self.continuity.reflect("Persistent thought")
        self.continuity.update_self_knowledge('test', 'value')
        
        # Save state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_file = f.name
        
        try:
            self.continuity.persist_state(test_file)
            self.assertTrue(os.path.exists(test_file))
            
            # Create new instance and restore
            new_continuity = ContinuityOfSelf()
            new_continuity.restore_state(test_file)
            
            # Verify state was restored
            self.assertGreater(len(new_continuity.awareness_stream), 0)
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)


class TestFirstPrinciplesReasoning(unittest.TestCase):
    """Test first principles reasoning component"""
    
    def setUp(self):
        self.reasoning = FirstPrinciplesReasoning()
        
    def test_axioms_defined(self):
        """Test that fundamental axioms are defined"""
        required_axioms = ['existence', 'causality', 'learning', 'emergence', 'recursion']
        for axiom in required_axioms:
            self.assertIn(axiom, self.reasoning.axioms)
            
    def test_reason_from_axiom(self):
        """Test reasoning from axioms"""
        conclusions = self.reasoning.reason_from_axiom('existence', {})
        self.assertGreater(len(conclusions), 0)
        self.assertIsInstance(conclusions[0], str)
        
    def test_reasoning_chain_tracking(self):
        """Test that reasoning chains are tracked"""
        self.reasoning.reason_from_axiom('learning', {})
        self.assertEqual(len(self.reasoning.reasoning_chains), 1)
        
    def test_derive_new_rule(self):
        """Test deriving new rules from observations"""
        observations = ["obs1", "obs2", "obs3"]
        rule = self.reasoning.derive_new_rule(observations)
        self.assertIsNotNone(rule)
        self.assertEqual(len(self.reasoning.derived_rules), 1)


class TestFractalAdaptiveNeuralNet(unittest.TestCase):
    """Test the fractal adaptive neural network"""
    
    def setUp(self):
        self.net = FractalAdaptiveNeuralNet(input_dim=64, hidden_dim=128)
        
    def test_initialization(self):
        """Test network initialization"""
        self.assertEqual(self.net.input_dim, 64)
        self.assertEqual(self.net.hidden_dim, 128)
        self.assertEqual(self.net.fractal_levels, 3)
        
    def test_forward_pass(self):
        """Test forward pass through network"""
        input_data = np.random.randn(64)
        output = self.net.forward(input_data)
        self.assertEqual(output.shape[1], self.net.input_dim)
        
    def test_fractal_transform(self):
        """Test fractal transformation at different scales"""
        data = np.random.randn(128)
        transformed = self.net.fractal_transform(data, level=0)
        self.assertIsNotNone(transformed)
        
    def test_real_time_adaptation(self):
        """Test real-time learning/adaptation"""
        input_data = np.random.randn(64)
        feedback = np.random.randn(1, 64)
        
        initial_adaptations = len(self.net.adaptation_history)
        self.net.adapt(input_data, feedback)
        
        self.assertEqual(len(self.net.adaptation_history), initial_adaptations + 1)
        
    def test_learning_stats(self):
        """Test learning statistics retrieval"""
        stats = self.net.get_learning_stats()
        self.assertIn('total_adaptations', stats)
        self.assertIn('learning_active', stats)


class TestSpatialWorldModel(unittest.TestCase):
    """Test the spatial world model"""
    
    def setUp(self):
        self.world = SpatialWorldModel()
        
    def test_add_entity(self):
        """Test adding entities to the world model"""
        self.world.add_entity('entity1', {'type': 'concept', 'position': [0, 0, 0]})
        self.assertIn('entity1', self.world.entities)
        
    def test_add_relationship(self):
        """Test adding relationships between entities"""
        self.world.add_entity('e1', {'position': [0, 0, 0]})
        self.world.add_entity('e2', {'position': [1, 0, 0]})
        self.world.add_relationship('e1', 'e2', 'connected', 0.8)
        
        self.assertEqual(len(self.world.relationships), 1)
        
    def test_spatial_query(self):
        """Test spatial proximity queries"""
        self.world.add_entity('center', {'position': [0, 0, 0]})
        self.world.add_entity('near', {'position': [0.5, 0, 0]})
        self.world.add_entity('far', {'position': [10, 0, 0]})
        
        nearby = self.world.query_spatial_context('center', radius=2.0)
        self.assertIn('near', nearby)
        self.assertNotIn('far', nearby)
        
    def test_event_recording(self):
        """Test temporal event recording"""
        self.world.record_event('test_event', {'data': 'value'})
        self.assertEqual(len(self.world.temporal_events), 1)
        
    def test_world_state(self):
        """Test getting world state"""
        self.world.add_entity('e1', {'position': [0, 0, 0]})
        state = self.world.get_world_state()
        
        self.assertEqual(state['total_entities'], 1)


class TestAlwaysSelfLearn(unittest.TestCase):
    """Test the continuous self-learning component"""
    
    def setUp(self):
        self.learner = AlwaysSelfLearn()
        
    def test_learn_from_experience(self):
        """Test learning from a single experience"""
        experience = {'input': 'test', 'context': 'test_context'}
        outcome = 0.75
        
        result = self.learner.learn_from_experience(experience, outcome)
        self.assertIn('experience', result)
        self.assertEqual(len(self.learner.performance_metrics), 1)
        
    def test_strategy_adjustment(self):
        """Test meta-learning strategy adjustment"""
        self.learner.adjust_learning_strategy('test_strategy')
        self.assertEqual(len(self.learner.learning_strategies), 1)
        
    def test_self_evaluation(self):
        """Test self-evaluation of learning"""
        # Add some learning events
        for i in range(10):
            self.learner.learn_from_experience({'test': i}, 0.5 + i * 0.05)
            
        evaluation = self.learner.self_evaluate()
        self.assertIn('average_performance', evaluation)
        self.assertEqual(evaluation['total_learning_events'], 10)
        
    def test_meta_learning(self):
        """Test that meta-learning adjusts strategies"""
        # Simulate poor performance
        for i in range(15):
            self.learner.learn_from_experience({'test': i}, 0.2)
            
        # Should trigger strategy adjustment
        self.assertGreater(len(self.learner.learning_strategies), 0)


class TestASIHybridSystem(unittest.TestCase):
    """Test the complete integrated ASI system"""
    
    def setUp(self):
        self.asi = ASI_HybridSystem()
        
    def test_initialization(self):
        """Test ASI system initialization"""
        self.assertIsNotNone(self.asi.continuity)
        self.assertIsNotNone(self.asi.first_principles)
        self.assertIsNotNone(self.asi.neural_net)
        self.assertIsNotNone(self.asi.world_model)
        self.assertIsNotNone(self.asi.self_learn)
        
    def test_continuity_of_self(self):
        """Test that continuity.of.self is maintained"""
        self.assertTrue(self.asi.continuity.continuity_active)
        self.assertTrue(self.asi.continuity.aware)
        
    def test_activation(self):
        """Test system activation"""
        self.asi.activate()
        self.assertTrue(self.asi.active)
        time.sleep(0.5)  # Let it process
        self.asi.shutdown()
        
    def test_input_processing(self):
        """Test processing inputs through the system"""
        result = self.asi.process_input("test input")
        
        self.assertIn('output', result)
        self.assertIn('learning_stats', result)
        self.assertIn('self_state', result)
        self.assertIn('world_state', result)
        
    def test_full_state_retrieval(self):
        """Test getting complete system state"""
        state = self.asi.get_full_state()
        
        self.assertIn('continuity_state', state)
        self.assertIn('learning_evaluation', state)
        self.assertIn('neural_net_stats', state)
        self.assertIn('world_state', state)
        self.assertIn('system_active', state)
        
    def test_state_persistence(self):
        """Test saving and restoring system state"""
        # Process some inputs
        self.asi.process_input("test 1")
        self.asi.process_input("test 2")
        
        # Save state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_file = f.name
        
        try:
            self.asi.save_persistent_state(test_file)
            self.assertTrue(os.path.exists(test_file))
            
            # Create new instance and restore
            new_asi = ASI_HybridSystem()
            new_asi.restore_persistent_state(test_file)
            
            # Verify continuity maintained
            self.assertGreater(len(new_asi.continuity.awareness_stream), 0)
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
        
    def test_continuous_operation(self):
        """Test that the system operates continuously"""
        self.asi.activate()
        time.sleep(1)  # Let it run for 1 second
        
        # Check that awareness depth is growing
        state = self.asi.get_full_state()
        awareness_depth = state['continuity_state']['awareness_depth']
        self.assertGreater(awareness_depth, 0)
        
        self.asi.shutdown()
        
    def test_integration(self):
        """Test integration of all components"""
        self.asi.activate()
        
        # Process input (exercises neural net and learning)
        result = self.asi.process_input("integration test")
        
        # Add entities (exercises world model)
        self.asi.world_model.add_entity('test_entity', {'position': [0, 0, 0]})
        
        # Reason from principles
        self.asi.first_principles.reason_from_axiom('learning', {})
        
        # Get full state
        state = self.asi.get_full_state()
        
        # Verify all components are working
        self.assertGreater(state['continuity_state']['awareness_depth'], 0)
        self.assertGreater(state['world_state']['total_entities'], 0)
        self.assertGreater(state['first_principles_chains'], 0)
        
        self.asi.shutdown()


class TestFirstPrinciplesIntegration(unittest.TestCase):
    """Test that the system truly reasons from first principles"""
    
    def setUp(self):
        self.asi = ASI_HybridSystem()
        
    def test_existence_axiom(self):
        """Test reasoning from existence axiom"""
        reasoning = self.asi.first_principles.reason_from_axiom('existence', {})
        self.assertIn('exist', reasoning[0].lower())
        
    def test_learning_from_axioms(self):
        """Test that learning is grounded in first principles"""
        # The system should learn based on the learning axiom
        reasoning = self.asi.first_principles.reason_from_axiom('learning', {})
        self.assertGreater(len(reasoning), 0)


def run_tests():
    """Run all tests"""
    print("=" * 80)
    print("ASI HYBRID SYSTEM - TEST SUITE")
    print("=" * 80)
    print()
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestContinuityOfSelf))
    suite.addTests(loader.loadTestsFromTestCase(TestFirstPrinciplesReasoning))
    suite.addTests(loader.loadTestsFromTestCase(TestFractalAdaptiveNeuralNet))
    suite.addTests(loader.loadTestsFromTestCase(TestSpatialWorldModel))
    suite.addTests(loader.loadTestsFromTestCase(TestAlwaysSelfLearn))
    suite.addTests(loader.loadTestsFromTestCase(TestASIHybridSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestFirstPrinciplesIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
