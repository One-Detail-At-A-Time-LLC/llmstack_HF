from llmstack.processors import Processor
from transformers import pipeline, AutoTokenizer
from sliding_window_evol_instruct_beam_search import SlidingWindowEvolInstructBeamSearch
from functools import lru_cache
import logging
from black import format_str, FileMode

class CodeWritingProcessor(Processor):
    def __init__(self, name):
        super().__init__(name)
        logging.basicConfig(level=logging.INFO)
        self.model = pipeline("text-generation", model="wizardcoder-python-34b-v1-AWQ")
        self.tokenizer = AutoTokenizer.from_pretrained("tree-sitter-java-125B")
        self.beam_search = SlidingWindowEvolInstructBeamSearch(beam_size=10, window_size=50)

    @lru_cache(maxsize=100)
    def process(self, input, code_generation_task: str, style: str, performance_optimization: bool, readability: bool, coding_style_guide: str):
        logging.info(f"Processing task: {code_generation_task}")
        if not input:
            raise ValueError("Input cannot be empty.")
        
        try:
            if code_generation_task == 'generate_code_from_description':
                return self.format_code(self.generate_code(input))
            elif code_generation_task == 'translate_code':
                return self.translate_code(input)
            elif code_generation_task == 'debug_code':
                return self.debug_code(input)
            else:
                raise ValueError("Invalid task type.")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None

    def format_code(self, code: str):
        return format_str(code, mode=FileMode())

    def generate_code(self, input):
        tokens = self.tokenizer(input, return_tensors="pt").input_ids
        generated_code = self.beam_search.search(tokens)
        return self.tokenizer.decode(generated_code, skip_special_tokens=True)

    def translate_code(self, input):
        # Your translation logic here.
        translated_code = "System.out.println('Hello, World!');"
        return translated_code

    def debug_code(self, input):
        # Your debugging logic here.
        debug_info = "No errors found."
        return debug_info

    def batch_process(self, inputs: list, task_type: str, style: str, performance: bool, readability: bool, style_guide: str):
        return [self.process(input, task_type, style, performance, readability, style_guide) for input in inputs]
