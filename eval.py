import json

import utils

import os
import openai
openai.api_key = utils.get_openai_api_key()

l = []
question = []
answer = []
# Now, let's read the JSON objects back from the file
with open('answer.json', 'r') as file:
    for line in file:
        # Convert the JSON string back to a dictionary
        json_obj = json.loads(line.strip())
        l.append(json_obj)
    print(l)
    for i in range(0, len(l), 2):
        print(i)
        question.append(l[i]['content'])
        answer.append(l [i+1]['content'])

print(question)
print(answer)

from trulens_eval import Tru
tru = Tru()

tru.reset_database()

records, feedback = tru.get_records_and_feedback(app_ids=[])

print(records.head())

# launches on http://localhost:8501/
tru.run_dashboard()