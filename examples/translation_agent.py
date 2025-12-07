from encompass import compile, branchpoint, record_score, effect
from encompass.std import action, early_stop
from core.llm import LanguageModel

def create_translation_agent(llm):
    @compile
    def translation_agent():
        # Choose function signature style
        options = ["Standard", "Template", "Trailing Return"]
        choice_idx = yield branchpoint("signature_style")
        
        style = options[choice_idx]
        
        code = ""
        if style == "Standard":
            code += "int add(int a, int b) {\n"
        elif style == "Template":
            code += "template <typename T>\nT add(T a, T b) {\n"
        elif style == "Trailing Return":
            code += "auto add(int a, int b) -> int {\n"
        
        # Choose body implementation
        body_options = ["Return expression", "Variable assignment"]
        body_idx = yield branchpoint("body_style")
        body_style = body_options[body_idx]
        
        if body_style == "Return expression":
            code += "    return a + b;\n"
        elif body_style == "Variable assignment":
            code += "    int result = a + b;\n    return result;\n"

        code += "}"
        
        # 3. Score the solution
        # We use the LLM to score the code
        # Since llm.score is async, we must yield an Effect to let the engine handle it
        score = yield effect(llm.score, code, "conciseness and modern C++ practices")
        
        # Boost score for preferred styles (demo logic)
        if style == "Template": score += 0.2
        if body_style == "Return expression": score += 0.1
        
        # 4. Record score and terminate
        yield record_score(score * 100)
        return code

    return translation_agent
