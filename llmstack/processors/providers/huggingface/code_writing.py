from llmstack.processors import Processor
from transformers import pipeline, AutoTokenizer
from llmstack.processors import code_writing  # I'm not sure what this import is for, so I've left it as is.
from sliding_window_evol_instruct_beam_search import SlidingWindowEvolInstructBeamSearch

class CodeWritingProcessor(Processor):
    def __init__(self, name):
        super().__init__(name)
        self.model = pipeline("text-generation", model="wizardcoder-python-34b-v1-AWQ")
        self.tokenizer = AutoTokenizer.from_pretrained("tree-sitter-java-125B")
        self.beam_search = SlidingWindowEvolInstructBeamSearch(beam_size=10, window_size=50)

    def process(self, input, code_generation_task: str, style: str, performance_optimization: bool, readability: bool,
                coding_style_guide: str):
        if not input:
            raise ValueError("Input cannot be empty.")
        
        try:
            if code_generation_task == 'generate_code_from_description':
                return self.generate_code(input)
            elif code_generation_task == 'translate_code':
                return self.translate_code(input)
            elif code_generation_task == 'debug_code':
                return self.debug_code(input)
            else:
                raise ValueError("Invalid task type.")
            
        except Exception as e:
            print(f"Error: {e}")
            return None

    def generate_code(self, input):
        tokens = self.tokenizer(input, return_tensors="pt").input_ids
        generated_code = self.beam_search.search(tokens)
        decoded_code = self.tokenizer.decode(generated_code, skip_special_tokens=True)
        return decoded_code

    def translate_code(self, input):
        # Add your logic to translate code here.
        translated_code = "System.out.println('Hello, World!');"
        return translated_code

    def debug_code(self, input):
        # Add your logic to debug code here.
        debug_info = "No errors found."
        return debug_info
