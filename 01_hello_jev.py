"""
01: One yes/no question. Decision Model returns a probability.
"""
import os
import time
from dotenv import load_dotenv

load_dotenv()
from langchain_typesafe import Noul, TypeSafeClassifier

# Managed Jav

classifier = TypeSafeClassifier(
    base_url="https://openrouter.ai/api",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="respan/span-01-lite:free",
)

# The Key ("urgent") is how you read the answer back.

QUESTIONS = {
    'urgent': Noul(instructions='Does this message need attention right now?')
}

STATE = 'The deploy failed twice and customers are good, production is not impacted, this can wait. Can someone look now?'

start = time.perf_counter()
response = classifier.invoke({'state': STATE, 'questions': QUESTIONS})
elapsed_ms = (time.perf_counter() - start) * 1000

urgency = response.nouls['urgent'].noul

print(f'model: ...{response.model}')
print(f'urgent: ...{urgency:.3f}..(probability the answer is yes)')
print(f'latency: ...{elapsed_ms:.0f} ms')

if urgency >= 0.8:
    print('-> paging the on-call engineer')
else:
    print('-> Open regular Support ticket')
