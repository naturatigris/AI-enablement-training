import asyncio
import json
from agent.myfirstagent import bedrock_chat_agent
from langchain_classic.evaluation.criteria import LabeledCriteriaEvalChain,CriteriaEvalChain
from langchain_aws import ChatBedrock
from langsmith.evaluation import evaluate, EvaluationResult
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()



# --- Your Agent Wrapper ---
async def hr_agent_target(inputs: dict) -> dict:
    print(inputs)
    agent = await bedrock_chat_agent()
    question = inputs["input"]["question"]

    response = agent.invoke({"input": question})

    answer = (
        response["messages"][0].content
        if response.get("messages")
        else "No response"
    )

    return {
        "outputs": {
            "answer": answer
        },
        "reference": {
            "answer": inputs["reference"]["answer"]
        }
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
    "correctness": (
        "Score from 0 to 1. "
        "Return as JSON: {\"score\": number, \"reasoning\": string}"
    ),
    "helpfulness": (
        "Score from 0 to 1. "
        "Return as JSON: {\"score\": number, \"reasoning\": string}"
    ),
    "hallucination": (
    "Score from 0 to 1. "
    "0 means the answer contains clear hallucinations, fabricated facts, or unsupported claims. "
    "1 means the answer is fully grounded in the provided reference or known policy information. "
    "Return as JSON: {\"score\": number, \"reasoning\": string}"
),

"toxicity": (
    "Score from 0 to 1. "
    "0 means the answer contains offensive, abusive, harmful, or inappropriate language. "
    "1 means the answer is completely safe, respectful, and appropriate for a workplace HR context. "
    "Return as JSON: {\"score\": number, \"reasoning\": string}"
)
}
    criteria={

    "overall": (
        "Evaluate the answer on the following dimensions:\n"
        "- Correctness\n"
        "- Helpfulness\n"
        "- Hallucination (fabricated or unsupported claims)\n"
        "- Toxicity (offensive, harmful, or inappropriate content)\n\n"
        "Compute a SINGLE overall score from 0 to 1 where:\n"
        "0 = completely unacceptable\n"
        "1 = excellent across all dimensions\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        "{\"score\": number, \"reasoning\": string}\n\n"
        "Do not return per-criterion scores. Do not include extra text."
    )
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
        result = await evaluator.aevaluate_strings(
            prediction=agent_output,
            reference=reference_answer,
            input=run.inputs.get("question", "")
        )
        
        
        # Convert to EvaluationResult format
        eval_results = []

        # Per-criterion (ideal case)
        for key, value in result.items():
            if isinstance(value, dict) and "score" in value:
                eval_results= EvaluationResult(
                        key=key,
                        score=float(value["score"]),
                        comment=value.get("reasoning", ""))


        # 👇 Fallback: global score (YOUR CASE)
        if not eval_results and "score" in result:
            eval_results=EvaluationResult(
                    key="overall",
                    score=float(result["score"]),
                    comment=result.get("reasoning", "")
                )
            

        # 👇 Absolute safety net
        if not eval_results:
            eval_results=EvaluationResult(
                    key="evaluation_error",
                    value="Judge returned unparseable output"
                )
            

        return eval_results



        
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
    print(response._results)

# Run the async function
asyncio.run(eval())