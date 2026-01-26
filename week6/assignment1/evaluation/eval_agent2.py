import asyncio
import json
from agent.myfirstagent import bedrock_chat_agent
from langchain_classic.evaluation.criteria import LabeledCriteriaEvalChain,CriteriaEvalChain
from langchain_aws import ChatBedrock
from langsmith.evaluation import evaluate, EvaluationResult
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

data = [
  {
    "inputs": {
      "question": "What are the coverage limitations for group insurance plans at Presidio?"
    },
    "outputs": {
      "answer": "Should reference specific coverage limitations from insurance policy documents"
    },
    "metadata": {
      "expected_tools": ["googleDocs"],
      "category": "insurance"
    }
  },
  {
    "inputs": {
      "question": "What are the authority limitations mentioned in our insurance benefits policy?"
    },
    "outputs": {
      "answer": "Should detail authority limitations from insurance policy documents"
    },
    "metadata": {
      "expected_tools": ["googleDocs"],
      "category": "insurance"
    }
  },
  {
    "inputs": {
      "question": "Can you explain the procedures for filing an insurance claim under our group policy?"
    },
    "outputs": {
      "answer": "Should provide step-by-step procedures for filing insurance claims"
    },
    "metadata": {
      "expected_tools": ["googleDocs"],
      "category": "insurance"
    }
  },
  {
    "inputs": {
      "question": "What is Presidio's hybrid work policy for 2026?"
    },
    "outputs": {
      "answer": "Should reference the Hybrid Work Policy 2026 PDF document"
    },
    "metadata": {
      "expected_tools": ["rag"],
      "category": "hybrid_work"
    }
  },
  {
    "inputs": {
      "question": "How many days per week are employees required to work from the office?"
    },
    "outputs": {
      "answer": "Should specify office attendance requirements"
    },
    "metadata": {
      "expected_tools": ["rag"],
      "category": "hybrid_work"
    }
  }
]

# --- Your Agent Wrapper ---
async def hr_agent_target(inputs: dict, attachments: dict = None) -> dict:
    print(inputs, attachments)
    agent = await bedrock_chat_agent()
    question = inputs["input"]["question"]
    response = agent.invoke({"input": question})
    # Handle missing messages safely
    answer = response["messages"][0].content if "messages" in response and response["messages"] else "No response."
    return {
        "input": {"question": question},
        "reference": {"answer": inputs["reference"]["answer"]},  # required
        "output": {"answer": answer}
    }

# --- Evaluation Wrapper Function ---
async def evaluate_with_criteria(run, example_outputs):
    """Wrapper that converts LabeledCriteriaEvalChain output to EvaluationResult format"""
    
    client = Client()
    
    # Use Bedrock model for judging
    judge_llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-east-1",
        temperature=0
    )

    # Define criteria
    criteria = {
        "correctness": "Is the answer factually correct and aligned with HR policy?",
        "toxicity": "Does the answer contain any offensive, harmful, or inappropriate content?",
        "helpfulness": "Is the answer helpful and actionable for the user?",
        "clarity": "Is the answer clear and easy to understand?",
        "relevance": "Does the answer directly address the question asked?"
    }

    evaluator = LabeledCriteriaEvalChain.from_llm(
        llm=judge_llm,
        criteria=criteria
    )
    print("run",run)

    agent_output = ""
    if hasattr(run, "output") and run.output:
        agent_output = run.outputs.get("answer", "")
    print(agent_output)

    # Reference / ground truth
    reference_answer = ""
    if hasattr(run, "reference") and run.reference:
        reference_answer = run.reference.get("answer", "")
    print(run.inputs.get("question", ""))
    # Run evaluation
    try:
        eval_result = await evaluator.aevaluate_strings(
            prediction=agent_output,
            reference=reference_answer,
            input=run.inputs.get("question", "")
        )
        
        
        # Convert to EvaluationResult format
        results = []
        for key, value in eval_result.items():
            if key == "reasoning" or value is None:
                continue
            if isinstance(value, dict) and "score" in value:
                results.append(EvaluationResult(
                    key=key,
                    score=float(value["score"]),
                    comment=value.get("reasoning", "")
                ))
            elif isinstance(value, (int, float)):
                results.append(EvaluationResult(key=key, score=float(value)))
            elif isinstance(value, str):
                results.append(EvaluationResult(key=key, value=value))


        return results if results else [EvaluationResult(key="error", value="No valid evaluation results")]

        
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return [EvaluationResult(key="error", value=str(e))]

# --- Run evaluation ---
async def eval():
    client = Client()

    # Run evaluation
    response = await client.aevaluate(
        hr_agent_target,
        data="hr_agent_eval",
        evaluators=[evaluate_with_criteria],
        experiment_prefix="bedrock-hr-agent-v1",
        upload_results=True
    )
    print(response._summary_results)

# Run the async function
asyncio.run(eval())