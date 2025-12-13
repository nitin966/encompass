"""
Translation agent for E2E tests.
"""
from encompass import compile, branchpoint, record_score, effect

@compile
def translation_agent(llm):
    # Depth 0: Signature Style
    sig_style = branchpoint(name="signature_style")
    
    # Depth 1: Body Style
    body_style = branchpoint(name="body_style")
    
    # Mock logic to determine score and result
    score = 0.0
    result = ""
    
    if sig_style == 0:
        result += "template <typename T>\nT add(T a, T b) {\n"
        score += 50.0
    else:
        result += "int add(int a, int b) {\n"
        score += 10.0
        
    if body_style == 0:
        result += "    return a + b;\n}"
        score += 60.0
    else:
        result += "    return a + b;\n}" # Same body for now
        score += 10.0
        
    record_score(score)
    return result

def create_translation_agent(llm):
    return lambda: translation_agent(llm)
