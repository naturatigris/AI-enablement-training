from langsmith import Client
from eval_dataset import eval_dataset
from dotenv import load_dotenv
load_dotenv()
client = Client()

DATASET_NAME = "Presidio HR Agent Evaluation"
\

# Get dataset by name or ID
# dataset = client.list_datasets(dataset_name="Presidio HR Agent Evaluation")  # Replace with actual ID or name
# for ds in dataset:
#     print(ds.id, ds.name, ds.created_at)

# Create dataset (idempotent)
dataset = client.create_dataset(
    dataset_name=DATASET_NAME,
    description="Evaluation dataset for Presidio HR assistant (RAG, tools, hallucination checks)"
)

# Upload examples
for example in eval_dataset:
    client.create_example(
        dataset_id=dataset.id,
        inputs={
            "question": example["question"]
        },
        outputs={
            "answer": example["expected_answer"]
        },
        metadata={
            "expected_tools": example["expected_tools"],
            "category": example["category"]
        }
    )

print(f"✅ Uploaded {len(eval_dataset)} examples to LangSmith dataset: {DATASET_NAME}")
