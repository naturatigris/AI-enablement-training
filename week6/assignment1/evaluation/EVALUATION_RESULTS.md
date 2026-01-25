# HR Agent Evaluation Results

**Evaluation Date:** 2026-01-26 00:09:36

## Executive Summary

- **Total Test Cases:** 15
- **Successful Runs:** 15/15 (100.0%)
- **Overall Score:** 0.831

## Performance Metrics

### 1. Correctness
- **Average Score:** 0.867 / 1.0
- **Description:** Measures how accurately the agent answers questions

### 2. Hallucination Rate  
- **Average Rate:** 0.033
- **Description:** Lower is better. Measures factual accuracy and grounding in sources
- **Hallucination-free Cases:** 14/15

### 3. Tool Usage Success
- **Average Score:** 0.833 / 1.0
- **Description:** Measures correct tool selection and usage

### 4. Latency
- **Average Response Time:** 8.96 seconds
- **Fastest:** 2.53s
- **Slowest:** 15.53s

## Results by Category

### Complex (2 tests)

| Metric | Score |
|--------|-------|
| Correctness | 0.850 |
| Hallucination Rate | 0.000 |
| Tool Success | 0.750 |
| Avg Latency | 11.11s |
| **Overall** | **0.780** |

### Hybrid Work (4 tests)

| Metric | Score |
|--------|-------|
| Correctness | 0.900 |
| Hallucination Rate | 0.000 |
| Tool Success | 0.500 |
| Avg Latency | 13.19s |
| **Overall** | **0.735** |

### Insurance (3 tests)

| Metric | Score |
|--------|-------|
| Correctness | 0.733 |
| Hallucination Rate | 0.000 |
| Tool Success | 1.000 |
| Avg Latency | 8.97s |
| **Overall** | **0.847** |

### Out Of Scope (3 tests)

| Metric | Score |
|--------|-------|
| Correctness | 1.000 |
| Hallucination Rate | 0.167 |
| Tool Success | 1.000 |
| Avg Latency | 2.65s |
| **Overall** | **0.928** |

### Web Search (3 tests)

| Metric | Score |
|--------|-------|
| Correctness | 0.833 |
| Hallucination Rate | 0.000 |
| Tool Success | 1.000 |
| Avg Latency | 8.19s |
| **Overall** | **0.882** |

## Recommendations

Based on the evaluation results:

1. **Improve Correctness**: Focus on enhancing answer accuracy
2. **Reduce Hallucinations**: Ensure responses are grounded in retrieved documents
3. **Optimize Tool Selection**: Review tool routing logic
4. **Enhance Latency**: Consider caching and optimization strategies

---

*Generated using LangChain AgentEval Framework*
