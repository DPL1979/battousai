"""
test_evolution.py — Tests for battousai.evolution
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.evolution import (
    CodeValidator, ValidationResult, CodeSandbox,
    AgentFactory, FitnessEvaluator, GeneticPool, AgentGenome,
)


SAFE_AGENT_CODE = '''
from battousai.agent import Agent

class EvolvedAgent(Agent):
    def think(self, tick):
        self.mem_write("tick", tick)
        self.yield_cpu()
'''

INVALID_AGENT_CODE = '''
this is not valid python syntax !!!! $$$$
'''

DANGEROUS_AGENT_CODE = '''
import os
x = 1 + 1
'''


class TestCodeValidator(unittest.TestCase):

    def setUp(self):
        self.validator = CodeValidator()

    def test_valid_code_passes(self):
        result = self.validator.validate(SAFE_AGENT_CODE)
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.valid)

    def test_invalid_syntax_fails(self):
        result = self.validator.validate(INVALID_AGENT_CODE)
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.valid)

    def test_validation_result_has_errors_for_invalid(self):
        result = self.validator.validate(INVALID_AGENT_CODE)
        self.assertIsInstance(result.errors, list)
        self.assertGreater(len(result.errors), 0)

    def test_validation_result_has_no_errors_for_valid(self):
        result = self.validator.validate(SAFE_AGENT_CODE)
        self.assertEqual(len(result.errors), 0)

    def test_dangerous_import_code_fails_validation(self):
        """Code with import os but no Agent class should fail (no Agent subclass)."""
        result = self.validator.validate(DANGEROUS_AGENT_CODE)
        # Should fail because no Agent subclass defined
        self.assertFalse(result.valid)


class TestCodeSandbox(unittest.TestCase):

    def setUp(self):
        self.sandbox = CodeSandbox(tick_limit=100)

    def test_safe_code_executes_successfully(self):
        code = "result = 2 + 2"
        success, stdout, stderr = self.sandbox.execute(code)
        self.assertTrue(success)

    def test_syntax_error_code_fails(self):
        code = "def broken(:"
        success, stdout, stderr = self.sandbox.execute(code)
        self.assertFalse(success)

    def test_print_output_captured(self):
        code = "print('sandbox test')"
        success, stdout, stderr = self.sandbox.execute(code)
        self.assertIn("sandbox test", stdout)

    def test_infinite_loop_blocked_by_tick_limit(self):
        """Sandbox must terminate infinite loops (returns False, not hang)."""
        sandbox = CodeSandbox(tick_limit=10)
        code = "while True: pass"
        success, stdout, stderr = sandbox.execute(code)
        self.assertFalse(success)

    def test_execute_returns_three_tuple(self):
        code = "x = 1"
        result = self.sandbox.execute(code)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)


class TestAgentFactory(unittest.TestCase):

    def setUp(self):
        self.factory = AgentFactory()

    def test_compile_valid_agent_code(self):
        agent_class = self.factory.compile_and_register(SAFE_AGENT_CODE)
        self.assertIsNotNone(agent_class)

    def test_compile_invalid_code_returns_none(self):
        agent_class = self.factory.compile_and_register(INVALID_AGENT_CODE)
        self.assertIsNone(agent_class)

    def test_compiled_agent_is_agent_subclass(self):
        from battousai.agent import Agent
        agent_class = self.factory.compile_and_register(SAFE_AGENT_CODE)
        if agent_class is not None:
            self.assertTrue(issubclass(agent_class, Agent))


class TestFitnessEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = FitnessEvaluator()

    def test_evaluate_returns_float(self):
        metrics = {
            "tasks_completed": 10,
            "messages_sent": 5,
            "errors": 0,
            "ticks_alive": 20,
        }
        score = self.evaluator.evaluate(metrics)
        self.assertIsInstance(score, float)

    def test_higher_tasks_gives_higher_fitness(self):
        m1 = {"tasks_completed": 1, "messages_sent": 0, "errors": 0, "ticks_alive": 10}
        m2 = {"tasks_completed": 100, "messages_sent": 0, "errors": 0, "ticks_alive": 10}
        score1 = self.evaluator.evaluate(m1)
        score2 = self.evaluator.evaluate(m2)
        self.assertGreater(score2, score1)

    def test_errors_reduce_fitness(self):
        m_clean = {"tasks_completed": 10, "messages_sent": 5, "error_rate": 0, "ticks_alive": 20}
        m_errors = {"tasks_completed": 10, "messages_sent": 5, "error_rate": 10, "ticks_alive": 20}
        score_clean = self.evaluator.evaluate(m_clean)
        score_errors = self.evaluator.evaluate(m_errors)
        self.assertGreater(score_clean, score_errors)


class TestGeneticPool(unittest.TestCase):

    def setUp(self):
        self.pool = GeneticPool(max_population=10, elite_fraction=0.2, mutation_rate=0.1)

    def test_add_genome_to_pool(self):
        genome = AgentGenome(source_code=SAFE_AGENT_CODE, config={}, fitness_score=0.5)
        self.pool.add(genome)
        self.assertEqual(self.pool.size(), 1)

    def test_top_k_returns_genomes(self):
        for i in range(5):
            genome = AgentGenome(
                source_code=SAFE_AGENT_CODE,
                config={},
                fitness_score=float(i),
            )
            self.pool.add(genome)
        selected = self.pool.top_k(3)
        self.assertEqual(len(selected), 3)

    def test_top_k_returns_highest_fitness(self):
        """top_k should return highest scoring genomes."""
        for i in range(5):
            genome = AgentGenome(
                source_code=SAFE_AGENT_CODE,
                config={},
                fitness_score=float(i),
            )
            self.pool.add(genome)
        selected = self.pool.top_k(3)
        scores = [g.fitness_score for g in selected]
        # Scores should include the highest
        self.assertEqual(max(scores), 4.0)

    def test_pool_respects_max_population(self):
        for i in range(15):
            genome = AgentGenome(
                source_code=SAFE_AGENT_CODE,
                config={},
                fitness_score=float(i),
            )
            self.pool.add(genome)
        self.assertLessEqual(self.pool.size(), 10)


if __name__ == "__main__":
    unittest.main()
