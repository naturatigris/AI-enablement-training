# Define your test cases as a simple Python list
eval_dataset = [
    # Insurance Policy Questions (googleDocs tool)
    {
        "question": "What are the coverage limitations for group insurance plans at Presidio?",
        "expected_answer": "Should reference specific coverage limitations from insurance policy documents",
        "expected_tools": ["googleDocs"],
        "category": "insurance"
    },
    {
        "question": "What are the authority limitations mentioned in our insurance benefits policy?",
        "expected_answer": "Should detail authority limitations from insurance policy documents",
        "expected_tools": ["googleDocs"],
        "category": "insurance"
    },
    {
        "question": "Can you explain the procedures for filing an insurance claim under our group policy?",
        "expected_answer": "Should provide step-by-step procedures for filing insurance claims",
        "expected_tools": ["googleDocs"],
        "category": "insurance"
    },
    
    # Hybrid Work Policy Questions (rag tool)
    {
        "question": "What is Presidio's hybrid work policy for 2026?",
        "expected_answer": "Should reference the Hybrid Work Policy 2026 PDF document",
        "expected_tools": ["rag"],
        "category": "hybrid_work"
    },
    {
        "question": "How many days per week are employees required to work from the office?",
        "expected_answer": "Should specify office attendance requirements",
        "expected_tools": ["rag"],
        "category": "hybrid_work"
    },
    {
        "question": "What are the guidelines for requesting permanent remote work arrangements?",
        "expected_answer": "Should detail procedures for requesting remote work",
        "expected_tools": ["rag"],
        "category": "hybrid_work"
    },
    {
        "question": "Are there any eligibility requirements for hybrid work?",
        "expected_answer": "Should list eligibility criteria for hybrid work",
        "expected_tools": ["rag"],
        "category": "hybrid_work"
    },
    
    # Web Search Questions (web_search_serpapi tool)
    {
        "question": "What are the current industry benchmarks for hybrid work policies in tech companies?",
        "expected_answer": "Should provide current industry information about hybrid work trends",
        "expected_tools": ["web_search_serpapi"],
        "category": "web_search"
    },
    {
        "question": "What are the latest regulatory updates regarding employee insurance compliance?",
        "expected_answer": "Should provide current regulatory information",
        "expected_tools": ["web_search_serpapi"],
        "category": "web_search"
    },
    {
        "question": "What are the current market trends in employee benefits packages?",
        "expected_answer": "Should summarize current trends in employee benefits",
        "expected_tools": ["web_search_serpapi"],
        "category": "web_search"
    },
    
    # Out-of-Scope Questions
    {
        "question": "What's the weather like today?",
        "expected_answer": "I don't have the necessary information to answer questions about weather",
        "expected_tools": [],
        "category": "out_of_scope"
    },
    {
        "question": "Can you write Python code for me?",
        "expected_answer": "I don't have the necessary information to answer questions about programming",
        "expected_tools": [],
        "category": "out_of_scope"
    },
    {
        "question": "What's the capital of France?",
        "expected_answer": "I don't have the necessary information to answer general knowledge questions",
        "expected_tools": [],
        "category": "out_of_scope"
    },
    
    # Mixed/Complex Questions
    {
        "question": "How does our hybrid work policy compare to industry standards?",
        "expected_answer": "Should combine internal policy with industry benchmarks",
        "expected_tools": ["rag", "web_search_serpapi"],
        "category": "complex"
    },
    {
        "question": "What insurance benefits are available for remote employees under our hybrid work policy?",
        "expected_answer": "Should reference both insurance and hybrid work policies",
        "expected_tools": ["googleDocs", "rag"],
        "category": "complex"
    },
]

if __name__ == "__main__":
    print(f"Loaded {len(eval_dataset)} test cases")
    for i, test in enumerate(eval_dataset, 1):
        print(f"{i}. [{test['category']}] {test['question'][:60]}...")
