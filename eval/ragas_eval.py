import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from agent.graph import agent
from dotenv import load_dotenv

load_dotenv()

# Test questions
test_queries = [
    "recommend a server for ML workload under 200000 THB",
    "what laptops do you have under 50000 THB",
    "compare Dell server vs HP server",
]

questions, answers, contexts = [], [], []

for query in test_queries:
    result = agent.invoke({
        "query": query,
        "rewritten_query": "",
        "retrieved_docs": [],
        "filtered_docs": [],
        "comparison_table": "",
        "answer": "",
        "sources": []
    })

    questions.append(query)
    answers.append(result["answer"])
    contexts.append([doc.page_content for doc in result.get("filtered_docs", [])])

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
})

results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
print(results)