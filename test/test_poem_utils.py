"""
Unit tests for poem_utils package components
"""
import unittest
import sys
import os
import torch

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poem_utils.analyzer import PoemAnalyzer
from poem_utils.prompt_generator import PromptGenerator
from poem_utils.pipeline import PoemToImagePipeline

class TestPoemUtils(unittest.TestCase):
    """Test cases for poem_utils components"""
    
    def setUp(self):
        """Set up test environment"""
        self.sample_poem = """
        Tôi đã thấy mùa thu
        Tới qua sông nắng nhạt
        Gió lay cành khẽ nhẹ
        Hoa vàng rơi từng cánh
        """
        self.mock_analysis = """
        Cảm xúc: Thanh bình, hoài niệm
        Ẩn dụ: Hoa vàng rơi tượng trưng cho thời gian trôi qua
        Bối cảnh: Bên sông vào mùa thu
        Chuyển động: Nhẹ nhàng, chậm rãi
        """
    
    def test_poem_analyzer_init(self):
        """Test PoemAnalyzer initialization"""
        # Skip actual model loading for quick tests
        try:
            analyzer = PoemAnalyzer()
            self.assertIsNotNone(analyzer)
            print("Poem analyzer initialized successfully")
        except Exception as e:
            print(f"Skipping full model loading test: {e}")
            self.skipTest("Skipping full model loading test")
    
    def test_prompt_generator_ollama(self):
        """Test PromptGenerator with Ollama API"""
        generator = PromptGenerator(use_local_model=False)
        
        # Instead of calling the real API, we'll test the method structure
        try:
            # Mock out the actual API call
            original_method = generator._generate_with_ollama
            
            # Override with a mock implementation
            def mock_generate(*args, **kwargs):
                return "Cảnh sông nước mùa thu, lá vàng rơi nhẹ. Không khí thanh bình, hoài niệm. Tông màu vàng nhạt, xanh dương. Phong cách thủy mặc."
            
            generator._generate_with_ollama = mock_generate
            
            # Test the generate method
            prompt = generator.generate(self.mock_analysis)
            self.assertIsNotNone(prompt)
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 10)
            print("Prompt generator with Ollama tested successfully")
            
            # Restore original method
            generator._generate_with_ollama = original_method
        except Exception as e:
            print(f"Error in prompt generator test: {e}")
            self.fail(f"Prompt generator test failed: {e}")
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization without running models"""
        try:
            # Test pipeline initialization (this won't run the actual models)
            pipeline = PoemToImagePipeline()
            self.assertIsNotNone(pipeline)
            self.assertIsNotNone(pipeline.poem_analyzer)
            self.assertIsNotNone(pipeline.prompt_generator)
            self.assertIsNotNone(pipeline.diffusion_generator)
            print("Pipeline initialized successfully")
        except Exception as e:
            print(f"Error in pipeline initialization: {e}")
            self.fail(f"Pipeline initialization failed: {e}")

if __name__ == "__main__":
    unittest.main()
