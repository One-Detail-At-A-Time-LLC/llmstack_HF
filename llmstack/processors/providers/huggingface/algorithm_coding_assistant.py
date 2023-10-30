from transformers import pipeline
from sliding_window_evol_instruct_beam_search import SlidingWindowEvolInstructBeamSearch
import ast

class AlgorithmCodingAssistantProcessor:
    def __init__(self, model_name="TheBloke/WizardMath-70B-V1.0-AWQ", beam_size=10, window_size=50):
        self.model = pipeline("text-generation", model=model_name)
        self.beam_search = SlidingWindowEvolInstructBeamSearch(beam_size=beam_size, window_size=window_size)

    def process(self, prompt: str, max_length: int):
        generated_text = self.beam_search.search(prompt)
        return {"generated_text": generated_text}

    def verify_code(self, code: str):
        try:
            ast.parse(code)
        except SyntaxError:
            return False

        try:
            exec(code)
        except Exception:
            return False
        return True

    def generate_code_snippet(self, prompt: str, max_length: int):
        data = self.process(prompt, max_length)
        generated_text = data.get("generated_text", "")
        
        lines = [line for line in generated_text.split("\n") if line]
        indented_lines = ["    " + line for line in lines]
        return "\n".join(indented_lines)

# Usage
if __name__ == "__main__":
    model = AlgorithmCodingAssistantProcessor()
    prompt = "Write a function to calculate the factorial of a number."
    max_length = 100
    code_snippet = model.generate_code_snippet(prompt, max_length)
    
    is_valid = model.verify_code(code_snippet)
    print(f"Generated Code Snippet:\n{code_snippet}")
    print(f"Is the code valid? {is_valid}")
