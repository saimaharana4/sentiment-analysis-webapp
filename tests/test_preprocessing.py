import unittest
from src.data_preprocessing import preprocess_text

class PreprocessingTestCase(unittest.TestCase):
    def test_preprocess_text(self):
        # Test case for the preprocess_text function
        
        # Example input and expected output
        input_text = "NLTK is a leading platform for building Python programs to work with human language data"
        expected_output = "nltk lead platform build python program work human languag data"

        # Call the preprocess function
        test_output = preprocess_text(input_text)

        # Check if the output is as expected
        self.assertEqual(test_output, expected_output, "The preprocess_text function does not return expected output")

if __name__ == '__main__':
    unittest.main()
