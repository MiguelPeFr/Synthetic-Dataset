from rich import print
import os
from openai import OpenAI
from datasets import Dataset, DatasetDict, load_dataset
import json

topic = "Unreal Engine Blueprints code"
n_subtopics = 2
n_questions = 2

# Configura tu clave API aquí
os.environ["nvdia-api"] = "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"  # Reemplaza con tu clave API real
api_key = os.environ["nvdia-api"]  # Accede a la variable de entorno por su nombre

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
)

# 1. Subtopics Generation

TOPIC_GENERATION_PROMPT_TEMPLATE = """
Dado un tema, genera una lista de {n_subtemas} subtemas que estén relacionados con el tema.

El tema es: {topic}

La lista debe ser sin numeros, y sin ninguna descripción de los subtopics. Los subtemas deben estar separados por una coma. No debe haber otro texto que la lista.
"""

def generate_subtopics(client, topic, n_subtopics):
    prompt = TOPIC_GENERATION_PROMPT_TEMPLATE.format(topic=topic, n_subtopics=n_subtopics)
    response = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[
            {"role": "user",
             "content": prompt}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    return response

responses = generate_subtopics(client, topic=topic, n_subtopics=n_subtopics)
print(responses.choices[0].message.content)

# 2. Questions Generation

QUESTION_PROMPT_TEMPLATE = """
Dado un tema, genera {n_questions} preguntas que puedan ser respondidas sobre ese tema

El tema es: {n_questions}

La lista debe ser sin numeros. La pregunta debe estar separada por un caracter de nueva linea.
"""

subtopic_list = responses.choices[0].message.content.split(",")
def generate_questions(client, sub_topic, n_questions):
    prompt = QUESTION_PROMPT_TEMPLATE.format(sub_topic=sub_topic, n_questions=n_questions)
    response = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[
            {"role": "user",
             "content": prompt}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def question_generator(client, subtopic_list, n_question):
    tasks = [generate_questions(client, subtopic, n_question) for subtopic in subtopic_list]
    question_list = tasks
    return question_list

question_list = question_generator(client, subtopic_list, n_questions)
print(question_list)

question_list_formatted = []
for question_set in question_list:
    question_list_formatted.extend([question.strip() for question in question_set.split("\n") if question])
len(question_list_formatted)

# 3. Responses Generation
RESPONSE_PROMPT_TEMPLATE = """
Dada una pregunta, genera 2 respuestas que puedan ser dadas a esta pregunta. La respuesta debe estar en formato lista.

La pregunsta es: {question}

La lista debe ser en el siguiente formato:

RESPUESTA A: Texto aquí de la respuesta A
RESPUESTA B: Texto aquí de la respuesta B
"""
def generate_responses(client, question):
    prompt = RESPONSE_PROMPT_TEMPLATE.format(question=question)
    response = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[
            {"role": "user",
             "content": prompt}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def response_generator(client, question_list):
    tasks = [generate_responses(client, question) for question in question_list]
    response_list = tasks
    return response_list

question_response_list = response_generator(client, question_list_formatted)
question_response_pair_list = []
for question, response_set in zip(question_list_formatted, question_response_list):
    question_response_pair_list.append(
        {
            "question": question,
            "responses": {
                "response_a": {"response": response_set.split("RESPONSE B:")[0].replace("RESPONSE A:", "").strip()},
                "response_b": {"response": response_set.split("RESPONSE B:")[-1].split("\n\n")[0].strip()}
            },
        }
    )

with open('synthetic_data.jsonl', 'w') as f:
    for item in question_response_pair_list:
        f.write(json.dumps(item))
        f.write('\n')

messages = [
    {
        "role": "user",
        "content": "Hello!"
    },
    {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
    },
]

response = client.chat.completions.create(
    model="nvidia/nemotron-4-340b-reward",
    messages=messages,
)

print(response)

print(response.choices[0].logprobs.content)

def get_scores_from_response(openai_response_template):
    logprobs = openai_response_template.choices[0].logprobs.content
    score_dict = {}
    for score in logprobs:
        score_dict[score.token] = score.logprob
    return score_dict

print(get_scores_from_response(response))

def get_response_and_scores(client, model, question, response_content):
    messages = [
        {
            "role": "user",
            "content": question
        },
        {
            "role": "assistant",
            "content": response_content
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    scores = get_scores_from_response(response)
    return scores

question_response_score_list = question_response_pair_list.copy()

def process_question_response_pairs(client, model, question_response_score_list):
    tasks = []
    for question_response_pair in question_response_score_list:
        question = question_response_pair["question"]

        task_a = get_response_and_scores(client, model, question, question_response_pair["responses"]["response_a"]["response"])
        task_b = get_response_and_scores(client, model, question, question_response_pair["responses"]["response_b"]["response"])

        tasks.append((task_a, question_response_pair, "response_a"))
        tasks.append((task_b, question_response_pair, "response_b"))

    results = [task[0] for task in tasks]

    for i, (result, task_info) in enumerate(zip(results, tasks)):
        _, question_response_pair, response_key = task_info
        question_response_pair["responses"][response_key].update(result)

process_question_response_pairs(client, "nvidia/nemotron-4-340b-reward", question_response_score_list)

threshold = 3.0

with open(f'synthetic_data_with_scores_filtered-{threshold}.jsonl', 'w') as f:
    for item in question_response_score_list:
        question = item["question"]
        response_a = item["responses"]["response_a"]
        response_b = item["responses"]["response_b"]
        response_a["question"] = question
        response_b["question"] = question
        if response_a["helpfulness"] < threshold and response_b["helpfulness"] < threshold:
            continue
        f.write(json.dumps(response_a))
        f.write('\n')
        f.write(json.dumps(response_b))
        f.write('\n')


with open(f'synthetic_data_with_scores_filtered-{threshold}.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
dataset = Dataset.from_list(data)
dataset_dict = DatasetDict({"train": dataset})
dataset_dict.push_to_hub("mervinpraison/preference-dataset-prep")
