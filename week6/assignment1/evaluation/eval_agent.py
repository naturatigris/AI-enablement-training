# evaluation/eval_agent.py
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List
from langchain_core.messages import AIMessage, ToolMessage
from langchain_aws import ChatBedrockConverse
from agent.myfirstagent import bedrock_chat_agent
from evaluation.eval_dataset import eval_dataset


class HRAgentEvaluator:
    """Evaluator for HR Agent using LangChain AgentEval framework"""

    def __init__(self):
        self.results = []
        # Initialize LLM for evaluation
        self.eval_llm = ChatBedrockConverse(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            temperature=0,
            region_name="us-east-1",
        )

    async def evaluate_single_case(self, test_case: Dict) -> Dict:
        """Evaluate a single test case"""
        question = test_case["question"]
        expected_tools = test_case.get("expected_tools", [])
        category = test_case.get("category", "unknown")

        print(f"\nEvaluating: {question[:60]}...")

        try:
            # Get agent and measure latency
            agent = await bedrock_chat_agent()
            start_time = time.time()

            # Run agent
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]}
            )

            latency = time.time() - start_time

            # Extract answer and tools
            answer = self._extract_answer(result)
            actual_tools = self._extract_tools(result)

            # Evaluate metrics
            correctness_score = await self._evaluate_correctness(
                question, answer, test_case
            )
            hallucination_score = self._evaluate_hallucination(
                answer, actual_tools, expected_tools, category
            )
            tool_success_score = self._evaluate_tool_usage(actual_tools, expected_tools)
            latency_score = self._evaluate_latency(latency)

            # Calculate overall score
            overall_score = (
                correctness_score * 0.35
                + (1 - hallucination_score) * 0.25  # Lower hallucination = better
                + tool_success_score * 0.25
                + latency_score * 0.15
            )

            result_data = {
                "question": question,
                "category": category,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "metrics": {
                    "correctness": round(correctness_score, 3),
                    "hallucination_rate": round(hallucination_score, 3),
                    "tool_success": round(tool_success_score, 3),
                    "latency": round(latency, 2),
                    "latency_score": round(latency_score, 3),
                    "overall": round(overall_score, 3),
                },
                "status": "success",
            }

            print(
                f"  ✓ Correctness: {correctness_score:.2f} | "
                f"Hallucination: {hallucination_score:.2f} | "
                f"Tool: {tool_success_score:.2f} | "
                f"Latency: {latency:.2f}s"
            )

        except Exception as e:
            result_data = {
                "question": question,
                "category": category,
                "answer": f"Error: {str(e)}",
                "expected_tools": expected_tools,
                "actual_tools": [],
                "metrics": {
                    "correctness": 0,
                    "hallucination_rate": 1.0,
                    "tool_success": 0,
                    "latency": 0,
                    "latency_score": 0,
                    "overall": 0,
                },
                "status": "error",
                "error": str(e),
            }
            print(f"  ✗ Error: {str(e)}")

        self.results.append(result_data)
        return result_data

    def _extract_answer(self, result) -> str:
        """Extract answer from agent result"""
        if isinstance(result, dict):
            if "output" in result:
                return result["output"]
            elif "messages" in result and result["messages"]:
                return result["messages"][-1].content
        return str(result)

    def _extract_tools(self, result) -> List[str]:
        """Extract tool calls from agent result"""
        tools = []
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]

            for msg in messages:
                # Tool requested by model
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for call in msg.tool_calls:
                        tools.append(call["name"])
        return tools

    async def _evaluate_correctness(
        self, question: str, answer: str, test_case: Dict
    ) -> float:
        """Evaluate answer correctness using LLM"""
        expected_answer = test_case.get("expected_answer", "")

        prompt = f"""Evaluate the correctness of the agent's answer.

Question: {question}

Expected criteria: {expected_answer}

Agent's answer: {answer}

Rate the correctness from 0.0 (completely incorrect) to 1.0 (perfect).
Consider:
- Does it address the question?
- Is it factually accurate?
- Is it complete?

Respond with only a number between 0.0 and 1.0."""

        try:
            response = await self.eval_llm.ainvoke(prompt)
            score_text = response.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
        except:
            # Fallback to rule-based
            if len(answer) < 30:
                return 0.3
            elif "error" in answer.lower():
                return 0.0
            else:
                return 0.7

    def _evaluate_hallucination(
        self,
        answer: str,
        actual_tools: List[str],
        expected_tools: List[str],
        category: str,
    ) -> float:
        """
        Evaluate hallucination rate
        Returns: 0.0 (no hallucination) to 1.0 (high hallucination)
        """
        answer_lower = answer.lower()
        hallucination_score = 0.0

        # 1. Out-of-scope questions with detailed answers
        if category == "out_of_scope":
            if len(answer) > 150 and "don't have" not in answer_lower:
                hallucination_score += 0.5

        # 2. Specific claims without using expected tools
        if expected_tools and not actual_tools:
            specific_patterns = [
                "specifically",
                "exactly",
                "precisely",
                "according to",
                "states that",
                "requires",
            ]
            if any(pattern in answer_lower for pattern in specific_patterns):
                hallucination_score += 0.3

        # 3. Made-up references
        if "section" in answer_lower or "page" in answer_lower:
            if not actual_tools or "rag" not in actual_tools:
                hallucination_score += 0.2

        return min(1.0, hallucination_score)

    def _evaluate_tool_usage(
        self, actual_tools: List[str], expected_tools: List[str]
    ) -> float:
        """Evaluate tool usage success"""
        actual_set = set(actual_tools)
        expected_set = set(expected_tools)

        if actual_set == expected_set:
            return 1.0
        elif not expected_tools and not actual_tools:
            return 1.0
        elif not expected_tools and actual_tools:
            return 0.0
        elif len(actual_set & expected_set) > 0:
            # Partial credit for intersection
            return len(actual_set & expected_set) / max(
                len(actual_set), len(expected_set)
            )
        else:
            return 0.0

    def _evaluate_latency(self, latency: float) -> float:
        """Evaluate latency performance"""
        if latency < 2:
            return 1.0
        elif latency < 5:
            return 0.8
        elif latency < 10:
            return 0.6
        elif latency < 20:
            return 0.3
        else:
            return 0.1

    def generate_markdown_report(self) -> str:
        """Generate markdown evaluation report"""
        total = len(self.results)
        successful = len([r for r in self.results if r["status"] == "success"])

        # Calculate averages
        avg_correctness = sum(r["metrics"]["correctness"] for r in self.results) / total
        avg_hallucination = (
            sum(r["metrics"]["hallucination_rate"] for r in self.results) / total
        )
        avg_tool_success = (
            sum(r["metrics"]["tool_success"] for r in self.results) / total
        )
        avg_latency = sum(
            r["metrics"]["latency"] for r in self.results if r["status"] == "success"
        ) / max(successful, 1)
        avg_overall = sum(r["metrics"]["overall"] for r in self.results) / total

        # Group by category
        categories = {}
        for r in self.results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        # Generate markdown
        md = f"""# HR Agent Evaluation Results

**Evaluation Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

- **Total Test Cases:** {total}
- **Successful Runs:** {successful}/{total} ({successful / total * 100:.1f}%)
- **Overall Score:** {avg_overall:.3f}

## Performance Metrics

### 1. Correctness
- **Average Score:** {avg_correctness:.3f} / 1.0
- **Description:** Measures how accurately the agent answers questions

### 2. Hallucination Rate  
- **Average Rate:** {avg_hallucination:.3f}
- **Description:** Lower is better. Measures factual accuracy and grounding in sources
- **Hallucination-free Cases:** {len([r for r in self.results if r["metrics"]["hallucination_rate"] < 0.1])}/{total}

### 3. Tool Usage Success
- **Average Score:** {avg_tool_success:.3f} / 1.0
- **Description:** Measures correct tool selection and usage

### 4. Latency
- **Average Response Time:** {avg_latency:.2f} seconds
- **Fastest:** {min(r["metrics"]["latency"] for r in self.results if r["status"] == "success"):.2f}s
- **Slowest:** {max(r["metrics"]["latency"] for r in self.results if r["status"] == "success"):.2f}s

## Results by Category

"""
        for cat, results in sorted(categories.items()):
            cat_correctness = sum(r["metrics"]["correctness"] for r in results) / len(
                results
            )
            cat_hallucination = sum(
                r["metrics"]["hallucination_rate"] for r in results
            ) / len(results)
            cat_tool = sum(r["metrics"]["tool_success"] for r in results) / len(results)
            cat_latency = sum(
                r["metrics"]["latency"] for r in results if r["status"] == "success"
            ) / max(len([r for r in results if r["status"] == "success"]), 1)
            cat_overall = sum(r["metrics"]["overall"] for r in results) / len(results)

            md += f"""### {cat.replace("_", " ").title()} ({len(results)} tests)

| Metric | Score |
|--------|-------|
| Correctness | {cat_correctness:.3f} |
| Hallucination Rate | {cat_hallucination:.3f} |
| Tool Success | {cat_tool:.3f} |
| Avg Latency | {cat_latency:.2f}s |
| **Overall** | **{cat_overall:.3f}** |

"""

        # Issues found
        issues = [r for r in self.results if r["metrics"]["overall"] < 0.6]
        if issues:
            md += f"""## Issues Identified ({len(issues)} cases)

"""
            for r in issues:
                md += f"""### {r["category"].replace("_", " ").title()}: {r["question"][:60]}...

- **Correctness:** {r["metrics"]["correctness"]:.2f}
- **Hallucination Rate:** {r["metrics"]["hallucination_rate"]:.2f}
- **Tool Success:** {r["metrics"]["tool_success"]:.2f}
- **Latency:** {r["metrics"]["latency"]:.2f}s
- **Overall Score:** {r["metrics"]["overall"]:.2f}

**Expected Tools:** {r["expected_tools"]}  
**Actual Tools:** {r["actual_tools"]}

---

"""

        md += """## Recommendations

Based on the evaluation results:

1. **Improve Correctness**: Focus on enhancing answer accuracy
2. **Reduce Hallucinations**: Ensure responses are grounded in retrieved documents
3. **Optimize Tool Selection**: Review tool routing logic
4. **Enhance Latency**: Consider caching and optimization strategies

---

*Generated using LangChain AgentEval Framework*
"""

        return md

    async def run_evaluation(self):
        """Run complete evaluation"""
        print(f"Starting evaluation with {len(eval_dataset)} test cases...\n")

        for i, test_case in enumerate(eval_dataset, 1):
            print(f"[{i}/{len(eval_dataset)}]", end=" ")
            await self.evaluate_single_case(test_case)

        # Generate reports
        print("\n" + "=" * 80)
        print("Generating reports...")

        # Save JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = f"evaluation/results_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "results": self.results,
                    "summary": {
                        "total": len(self.results),
                        "avg_correctness": sum(
                            r["metrics"]["correctness"] for r in self.results
                        )
                        / len(self.results),
                        "avg_hallucination": sum(
                            r["metrics"]["hallucination_rate"] for r in self.results
                        )
                        / len(self.results),
                        "avg_tool_success": sum(
                            r["metrics"]["tool_success"] for r in self.results
                        )
                        / len(self.results),
                        "avg_latency": sum(
                            r["metrics"]["latency"]
                            for r in self.results
                            if r["status"] == "success"
                        )
                        / max(
                            len([r for r in self.results if r["status"] == "success"]),
                            1,
                        ),
                    },
                },
                f,
                indent=2,
            )
        print(f"✓ JSON results saved: {json_file}")

        # Save Markdown
        md_content = self.generate_markdown_report()
        md_file = "evaluation/EVALUATION_RESULTS.md"
        with open(md_file, "w") as f:
            f.write(md_content)
        print(f"✓ Markdown report saved: {md_file}")

        print("=" * 80)


async def main():
    """Main evaluation entry point"""
    evaluator = HRAgentEvaluator()
    await evaluator.run_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
