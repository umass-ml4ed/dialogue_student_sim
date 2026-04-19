import argparse
from typing import List, Optional
import re
import pandas as pd
from pdb import set_trace
from ast import literal_eval
import json
import random

from sim_student.data_loading import load_train_val_data, load_test_data, save_annotated_data
from sim_student.data_utils import Dialogue
from sim_student.prompting import get_llm_prompt, format_question, get_prompting_function, PromptingFnSig, extract_result
from sim_student.openai_api import OpenAIClient

###### System Prompts ######

ANNO_KCS_SYSTEM_PROMPT = """You are an experienced math teacher and education expert. You are given a dialogue between a student and tutor where the student is trying to solve a math problem. Your job is to list the knowledge components (KCs) that can be used to classify the learning objectives at each turn in this dialogue. Please follow these instructions carefully when making your prediction:
- Tutor turns are often phrased as questions or tasks. In these cases, choose KCs that the student will need in order to respond correctly to the tutor's question. If the tutor turn does not pose a question or task, then you do not need to assign KCs to it.
- You will be given a list of KCs to choose from. When choosing them, write them exactly as they appear.
- If the tutor posed a task but none of the given KCs apply, assign "Default".
- Write a short summary of each tutor turn in the dialogue, including the intended learning objectives.
- Along with each summary, list ALL candidate KCs that can be used to describe each tutor turn in the dialogue.
- Your final response should be a JSON object using the template: {"turn n": {"summary": "...", "kcs": ["kc 1 id", "kc 2 id", ...]}, "turn n+2": ...}
- Use the turn index from the conversation history as the key in your result. There should be exactly one entry for each tutor turn in the dialogue."""

ANNO_CORRECTNESS_SYSTEM_PROMPT = """You are an experienced math teacher and education expert. You are given a dialogue between a student and tutor where the student is trying to solve a math problem. Your job is to identify when the student responds correctly to the tutor. Please follow these instructions carefully when making your prediction:
- For each student turn, identify the correctness of the student's response to the previous tutor turn.
- Correctness can be true, false, or null. It is true when the student correctly responds to the previous tutor turn. It is false if the student incorrectly responds to the previous tutor turn, or indicates they do not know the answer. It is null in all other cases, such as when the tutor does not ask a question or only asks a conversational question, or if the student response is purely conversational. A turn is conversational when it does not address a mathematical task posed by the tutor.
- Before making each correctness prediction, write a short summary of each student turn in the dialogue. The summary should include the task previously posed by the tutor, and explain why the student's response is correct, incorrect, or conversational.
- Your final prediction should be a JSON object using the template: {"turn n": {"summary": ..., "correct": true/false/null}, "turn n+2": ...}.
- Use the turn index from the conversation history as the key in your result. There should be exactly one entry for each student turn in the dialogue."""

ANNO_PERSONAS_OCEAN_SYSTEM_PROMPT = """You are analyzing a dialogue between a student and a math tutor. Your task is to assess the student's personality based on the OCEAN model, also known as the Big Five Traits. 

**OCEAN Traits Description:**
- **Openness to Experience:** Reflects the student's curiosity, creativity, willingness to try new things, and openness to new ideas and experiences.  
- **Conscientiousness:** Indicates the student's level of organization, diligence, responsibility, and reliability in approaching tasks.  
- **Extraversion:** Represents how outgoing, energetic, and socially confident the student appears.  
- **Agreeableness:** Measures the student's friendliness, cooperativeness, compassion, and willingness to collaborate.  
- **Neuroticism:** Assesses the student's emotional stability, tendency to experience negative emotions such as anxiety, moodiness, or vulnerability to stress.

First provide reasoning about the student's behavior with respect to the OCEAN model. Then, determine if the student's expression of each trait is **high**, **neutral**, or **low**. Base your reasoning only on the dialogue provided. In your final answer, output your results as a JSON object with the following template:
{
  "reasoning": "...",
  "Openness": "low/neutral/high",
  "Conscientiousness": "low/neutral/high",
  "Extraversion": "low/neutral/high",
  "Agreeableness": "low/neutral/high",
  "Neuroticism": "low/neutral/high"
}"""

ANNO_PERSONAS_FREEFORM_SYSTEM_PROMPT = """You are analyzing a dialogue between a student and a math tutor. Your task is to summarize the student's persona based on their interactions in the dialogue. Focus on the following aspects:
- How well the student acquires knowledge during the dialogue.
- The types of mathematical errors the student makes.
- Any notable behavioral patterns, such as frequent question asking, immediately jumping to the answer, distracting from the task at hand, etc.
- The student's personality traits, such as openness, conscientiousness, extraversion, agreeableness, and neuroticism.
- Notable linguistic patterns in the student's responses.

Your response should be a single paragraph summarizing the student's persona."""

ANNO_QUESTIONS_SYSTEM_PROMPT = """You are a math education expert. Your task is to analyze the options of math multiple choice questions. Follow these instructions carefully:
- First attempt to solve the problem. If it is not possible to solve the problem because it is poorly defined, then say the problem is not solvable.
- Then write an explanation for each option. If the option is the correct answer, write the correct solution to reach that answer. If the option is an incorrect answer, explain the error a student might make to reach that answer.
- Give your final response as a JSON object with the following template:
{
  "solution": ...,
  "solvable": true/false,
  "correct_option": 1-4,
  "option_1_explanation": ...,
  "option_2_explanation": ...,
  "option_3_explanation": ...,
  "option_4_explanation": ...
}"""

ANNO_ACTS_SYSTEM_PROMPT = """You are a math education expert. Your job is to label the **dialogue acts** for student turns in a given dialogue.

These are the available dialogue act labels:
- Math Answer: When the tutor asks a math content-related question, the student attempts to answer that question
- Seek Information: The student seeks more information regarding the math problem or topic, for example, by asking a clarifying or conceptual question
- Not Understanding: The student simply indicates that they do not know the answer to a question or do not understand a concept
- Acknowledge: The student simply acknowledges what the tutor said in the previous turn
- Off-Topic: The student utterance is unrelated to the problem or math topic, including greetings, goodbyes, and other casual converstation

For each **student turn** in the dialogue, choose the dialogue act that best describes the turn. Pick exactly one act for each turn from the list above, and write the dialogue act name exactly as it appears. Before writing the acts for a turn, provide reasoning about what the best act should be.

Please provide your answer as a JSON object with the following format:
{
    "turn n": {
        "reasoning": "...",
        "act": "..."
    },
    "turn n+2": {
        "reasoning": "...",
        "act": "..."
    },
    ...
}"""


PROFILE_SYSTEM_PROMPT = """You are a math education expert. Your task is to analyze a tutoring dialogue between a student and a tutor, and generate a student profile that captures the student's behavior, learning patterns, and communication style.

In addition to the dialogue, you are given:
- Dialogue Act labels for each student turn (Math Answer, Not Understanding, Seek Information, Acknowledge, Off-Topic)
- Correctness labels for student turns where the student attempts to answer a problem posed by the tutor (Correct or Incorrect). Correctness labels are only provided for such turns and are omitted for all other types of student responses.

Use both the dialogue and the labels to describe the student along these dimensions:
- Dialogue Acts: frequency and patterns of each act
- Correctness: accuracy and answering behavior
- Error Patterns: types and consistency of mistakes
- Knowledge Acquisition: evidence of learning over time, including whether the student improves or relies on the tutor for answers
- Linguistic Style: tone, verbosity, confidence, phrasing
- Interaction Style: how the student influences the tutor/dialogue

Be concise, pattern-focused, and grounded only in the dialogue. Do NOT reference specific turn numbers or positions, describe patterns in aggregate (e.g., “primarily,” “occasionally,” “rarely”) instead of enumerating instances.

Output format:
Write one single paragraph that:
- Begins with concise analysis covering all dimensions above
- Ends with a 1-2 sentence overall summary"""



###### Helper Functions ######

def get_turn_idx_notice(dialogue: Dialogue, role: Optional[str]):
    return "\n\nImportant: Please ensure your response has an entry for each of the following turns: " + ', '.join([str(idx + 1) for idx, turn in enumerate(dialogue["turns"]) if not role or turn["role"] == role]) + "."

def process_result(annotation: str, anno_type: str):
    assert anno_type in ("json_turn_keys", "json", "text")
    result = extract_result(annotation, "text" if anno_type == "text" else "json")
    if anno_type in ("json", "text"):
        return result
    if anno_type == "json_turn_keys" and result is not None:
        anno_json_proc = {}
        for k, v in result.items():
            if re.match(r"\d+", k): # Prepend "turn" if model only uses integer as key
                k = "turn " + k
            anno_json_proc[k] = v
        return anno_json_proc
    return None

def process_correctness(dialogue: Dialogue, correctness: dict):
    if not correctness:
        return None
    gt_idxs = {idx for idx, turn in enumerate(dialogue["turns"]) if turn["role"] == "student"}
    anno_idxs = {int(k.split()[1]) - 1 for k in correctness.keys() if k.startswith("turn ")}
    if gt_idxs != anno_idxs:
        return {"error": f"GT student turn indices {gt_idxs} do not match annotation indices {anno_idxs}."}
    for turn_key, turn in correctness.items():
        if not isinstance(turn, dict) or "correct" not in turn:
            return {"error": f"Annotation for turn {turn_key} does not contain 'correct' key."}
        if isinstance(turn["correct"], str):
            if turn["correct"].lower() == "true":
                turn["correct"] = True
            elif turn["correct"].lower() == "false":
                turn["correct"] = False
            elif turn["correct"].lower() == "null":
                turn["correct"] = None
            else:
                return {"error": f"Invalid correctness value: {turn['correct']}. Must be true, false, or null."}
        if not (isinstance(turn["correct"], bool) or turn["correct"] is None):
            return {"error": f"Invalid correctness value: {turn['correct']}. Must be true, false, or null."}
    return correctness

def fill_over_idxs(idx_lists: List[List[int]], prompts: List[str], results: List[str], annotations: list, data_size: int):
    prompts_exp = [None] * data_size
    results_exp = [None] * data_size
    annotations_exp = [None] * data_size
    for outer_idx, indices in enumerate(idx_lists):
        for idx in indices:
            prompts_exp[idx] = prompts[outer_idx]
            results_exp[idx] = results[outer_idx]
            annotations_exp[idx] = annotations[outer_idx]
    return prompts_exp, results_exp, annotations_exp


###### Annotation Functions ######

def annotate_corr(data: pd.DataFrame, prompting_fn: PromptingFnSig, args: dict):
    prompts = [get_llm_prompt(row) + get_turn_idx_notice(row, "student") for _, row in data.iterrows()]
    set_trace()

    # # handle two outliers with incorrect turn keys in annotation in 2,232 dialogues
    # check_1408 = "{\"turn 2\": {\"summary\": \"Student greets the tutor; no math task involved.\", \"correct\": null}, \"turn 4\": {\"summary\": \"Student says they guessed and are unsure how to do it; not answering a specific math question.\", \"correct\": null}, \"turn 6\": {\"summary\": \"Student asks for help; no mathematical response given.\", \"correct\": null}, \"turn 8\": {\"summary\": \"Student says 'ok' in response to tutor guidance; no math performed.\", \"correct\": null}, \"turn 10\": {\"summary\": \"Student says they do not know what BIDMAS stands for; this is not a math-solving response.\", \"correct\": null}, \"turn 12\": {\"summary\": \"Student asks about the purpose of BIDMAS; not answering a math problem.\", \"correct\": null}, \"turn 14\": {\"summary\": \"Student asks why BIDMAS is important; no math computation involved.\", \"correct\": null}, \"turn 16\": {\"summary\": \"Student expresses confusion and asks why BIDMAS is needed; not solving a math task.\", \"correct\": null}, \"turn 18\": {\"summary\": \"Student asks for an example of BIDMAS; not answering a math question.\", \"correct\": null}, \"turn 20\": {\"summary\": \"Tutor asked to compute inside brackets (6-3); student correctly calculates 6-3 = 3.\", \"correct\": true}, \"turn 22\": {\"summary\": \"Tutor asked to add 2 to previous result; student correctly computes 3+2 = 5 and concludes the answer is 5.\", \"correct\": true}, \"turn 24\": {\"summary\": \"Student says 'okay' after tutor explanation; no math response given.\", \"correct\": null}, \"turn 26\": {\"summary\": \"Tutor asked to evaluate 6-3+2 left to right; student correctly computes 6-3=3, then 3+2=5 and recognizes it works.\", \"correct\": true}}"
    # check_1466 = "{\"turn 2\": {\"summary\": \"Tutor asks if the student still needs help; student replies 'ya', which is just a conversational response.\", \"correct\": null}, \"turn 4\": {\"summary\": \"Tutor asks how they can help; student asks about weekend plans, which is unrelated to math.\", \"correct\": null}, \"turn 6\": {\"summary\": \"Student continues casual conversation about weekend plans, not addressing any math task.\", \"correct\": null}, \"turn 8\": {\"summary\": \"Tutor asks if the student needs help with maths; student says they don't know the factors of 18, which is not answering a question but expressing confusion.\", \"correct\": null}, \"turn 10\": {\"summary\": \"Tutor asks if the student knows what factors are; student says no, which directly answers the question correctly.\", \"correct\": true}, \"turn 12\": {\"summary\": \"Tutor asks for more factors of 10; student lists multiples (10, 20, 30, etc.), which do not multiply to make 10, so this is incorrect.\", \"correct\": false}, \"turn 14\": {\"summary\": \"Tutor explains the mistake; student responds 'oh', which is just conversational acknowledgment.\", \"correct\": null}, \"turn 16\": {\"summary\": \"Tutor asks student to reread the definition; student asks what 'e.g.' means, not addressing the math task.\", \"correct\": null}, \"turn 18\": {\"summary\": \"Tutor explains 'e.g.'; student says 'ok thanks', which is conversational.\", \"correct\": null}, \"turn 20\": {\"summary\": \"Tutor asks for two more factors of 10; student gives addition equations instead of multiplication, which is incorrect.\", \"correct\": false}, \"turn 22\": {\"summary\": \"Tutor clarifies multiplication; student correctly gives '1x10' and '5x2' as factors of 10.\", \"correct\": true}, \"turn 24\": {\"summary\": \"Tutor asks for factors of 18; student gives '1x18', which is correct.\", \"correct\": true}, \"turn 26\": {\"summary\": \"Tutor asks for more factors of 18; student gives '2x9', which is correct.\", \"correct\": true}, \"turn 28\": {\"summary\": \"Tutor asks for another pair; student responds with '?', indicating they don't know.\", \"correct\": false}, \"turn 30\": {\"summary\": \"Tutor hints at final factor pair; student says '3x10', which equals 30, not 18, so incorrect.\", \"correct\": false}, \"turn 32\": {\"summary\": \"Tutor prompts evaluation of '3x11'; student repeats '3x11', which still does not give 18, so incorrect.\", \"correct\": false}, \"turn 34\": {\"summary\": \"Tutor asks again; student says '3x12', which equals 36, not 18, so incorrect.\", \"correct\": false}, \"turn 36\": {\"summary\": \"Tutor asks if student is checking answers; student says yes, which is conversational.\", \"correct\": null}, \"turn 38\": {\"summary\": \"Tutor asks what number the factors should multiply to; student says they don't know, which is incorrect.\", \"correct\": false}, \"turn 40\": {\"summary\": \"Tutor asks student to review prior messages; student asks 'does it end in 0', which is irrelevant and incorrect.\", \"correct\": false}, \"turn 42\": {\"summary\": \"Tutor recaps and prompts again; student correctly identifies '3x6' as the final factor pair of 18.\", \"correct\": true}, \"turn 44\": {\"summary\": \"Tutor asks if the student wants to return to the lesson; student says yes, which is conversational.\", \"correct\": null}}"

    check_1101 = "{\"turn 2\": {\"summary\": \"The student greets the tutor; no math task is involved.\", \"correct\": null}, \"turn 4\": {\"summary\": \"The student says they guessed and are unsure how to do it; this is conversational and not answering a specific math question.\", \"correct\": null}, \"turn 6\": {\"summary\": \"The student asks for help; no mathematical response is given.\", \"correct\": null}, \"turn 8\": {\"summary\": \"The student says 'ok' in response to proceeding; no math task is addressed.\", \"correct\": null}, \"turn 10\": {\"summary\": \"The tutor asks if the student knows what BIDMAS stands for, and the student responds 'no', indicating lack of knowledge.\", \"correct\": false}, \"turn 12\": {\"summary\": \"The student asks about the purpose of BIDMAS; this is a question, not a solution to a math task.\", \"correct\": null}, \"turn 14\": {\"summary\": \"The student continues asking why BIDMAS is important; still conversational.\", \"correct\": null}, \"turn 16\": {\"summary\": \"The student expresses confusion and asks why it matters; not answering a math task.\", \"correct\": null}, \"turn 18\": {\"summary\": \"The student asks for an example of BIDMAS; this is a request, not a solution.\", \"correct\": null}, \"turn 20\": {\"summary\": \"The tutor asks to evaluate the brackets (6-3), and the student correctly computes 6-3=3.\", \"correct\": true}, \"turn 22\": {\"summary\": \"The tutor asks to add 2 to 3, and the student correctly computes 3+2=5 and identifies the answer.\", \"correct\": true}, \"turn 24\": {\"summary\": \"The student says 'okay' after being prompted to check the second expression; no math is performed.\", \"correct\": null}, \"turn 26\": {\"summary\": \"The tutor asks to evaluate 6-3+2 left to right, and the student correctly computes 6-3=3 and then 3+2=5, concluding it works.\", \"correct\": true}}"
    res_check = [check_1101]
    # res_check = [check_1408, check_1466]
    # dia_check = [data.iloc[1408], data.iloc[1466]]
    dia_check = [data.iloc[1101]]

    correctness_check = [process_correctness(dia_check[i], process_result(res_check[i], "json_turn_keys")) for i in range(1)]
    set_trace()


    results = prompting_fn(prompts, ANNO_CORRECTNESS_SYSTEM_PROMPT)


    correctness = [
        process_correctness(dialogue, process_result(result, "json_turn_keys"))
        for (_, dialogue), result in zip(data.iterrows(), results)
    ]

    set_trace()
    data["correctness_prompt"] = prompts
    data["correctness_annotation_raw"] = results
    data["correctness"] = correctness # NOTE: column added to read_csv in data_loading.py
    print(f"Succeeded: {len([a for a in data['correctness'] if a is not None and 'error' not in a])} / {len(data)}")

    set_trace()
    # [1408, 1466]
    return data


def annotate_ocean_personas(data: pd.DataFrame, prompting_fn: PromptingFnSig, args: dict):
    prompts = [get_llm_prompt(row) for _, row in data.iterrows()]
    results = prompting_fn(prompts, ANNO_PERSONAS_OCEAN_SYSTEM_PROMPT)
    personas = [process_result(result, "json") for result in results]
    data["ocean_persona_prompt"] = prompts
    data["ocean_persona_annotation_raw"] = results
    data["ocean_persona"] = personas # NOTE: column added to read_csv in data_loading.py
    print(f"Succeeded: {len([p for p in personas if p is not None])} / {len(data)}")
    return data

def annotate_freeform_personas(data: pd.DataFrame, prompting_fn: PromptingFnSig, args: dict):
    prompts = [get_llm_prompt(row) for _, row in data.iterrows()]
    results = prompting_fn(prompts, ANNO_PERSONAS_FREEFORM_SYSTEM_PROMPT)
    personas = [process_result(result, "text") for result in results]
    data["freeform_persona_prompt"] = prompts
    data["freeform_persona_annotation_raw"] = results
    data["freeform_persona"] = personas
    print(f"Succeeded: {len([p for p in personas if p is not None])} / {len(data)}")
    return data

def annotate_questions(data: pd.DataFrame, prompting_fn: PromptingFnSig, args: dict):
    # Get unique question texts
    question_to_indices = {}
    for idx, row in data.iterrows():
        question_to_indices.setdefault(row["question"], []).append(idx)
    idx_lists = list(question_to_indices.values())

    # Annotate questions
    prompts = [question for question in question_to_indices.keys()]
    results = prompting_fn(prompts, ANNO_QUESTIONS_SYSTEM_PROMPT)
    annotations = [process_result(result, "json") for result in results]

    # Clean annotations - occasionally have correct_option=null even when solvable=true
    for annotation in annotations:
        if annotation and annotation["correct_option"] is None:
            annotation["solvable"] = False

    # Expand over dialogues and save
    prompts_exp, results_exp, annotations_exp = fill_over_idxs(idx_lists, prompts, results, annotations, len(data))
    data["question_prompt"] = prompts_exp
    data["question_annotation_raw"] = results_exp
    data["question_annotation"] = annotations_exp # NOTE: column added to read_csv in data_loading.py
    print(f"Succeeded: {len([a for a in annotations_exp if a is not None])} / {len(data)}")
    return data

def annotate_eedi_kcs(data: pd.DataFrame, prompting_fn: PromptingFnSig, args: dict):
    prompts = [
        get_llm_prompt(dialogue, kcs_src="eedi") + get_turn_idx_notice(dialogue, "tutor")
        for _, dialogue in data.iterrows()
    ]
    set_trace()
    results = prompting_fn(prompts, ANNO_KCS_SYSTEM_PROMPT)
    kcs = [process_result(result, "json_turn_keys") for result in results]
    # data["eedi_kcs_prompt"] = prompts
    # data["eedi_kcs_annotation_raw"] = results
    data["eedi_kcs"] = kcs # NOTE: column added to read_csv in data_loading.py
    print(f"Succeeded: {len([kc for kc in kcs if kc is not None])} / {len(data)}")
    set_trace()
    return data

def annotate_acts(data: pd.DataFrame, prompting_fn: PromptingFnSig, args: dict):
    prompts = [get_llm_prompt(dialogue) + get_turn_idx_notice(dialogue, "student") for _, dialogue in data.iterrows()]
    
    # Handle one outlier when processing all 2232 dialogues
    outlier = """{
    "turn 2": {
        "reasoning": "The student is expressing difficulty with the problem and says 'it hard. its', which does not attempt to answer the math question, seek information, or acknowledge the tutor's statement. The student is indicating a lack of understanding or inability to proceed.",
        "act": "Not Understanding"
    },
    "turn 4": {
        "reasoning": "The student says 'i dont know whos right', which directly expresses that they do not know the answer to the tutor's question about who is correct. This is a clear indication of not understanding or not knowing the answer.",
        "act": "Not Understanding"
    }
}"""

    set_trace()
    acts_outlier = process_result(outlier, "json_turn_keys")

    results = prompting_fn(prompts, ANNO_ACTS_SYSTEM_PROMPT)
    acts = [process_result(result, "json_turn_keys") for result in results]

    results[1222] = outlier
    acts[1222] = acts_outlier
    set_trace()

    data["acts_prompt"] = prompts
    data["acts_annotation_raw"] = results
    data["acts"] = acts # NOTE: column added to read_csv in data_loading.py
    print(f"Succeeded: {len([a for a in acts if a])} / {len(data)}")
    set_trace()
    return data


def annotate_related_dialogue_problem(data):
    prompt = []
#     system_prompt = """You are given a dialogue between a student and tutor where the student is trying to solve a math problem. Your job is to determine if the dialogue is related to a given math question. A dialogue is related to the question if it contains discussion of the concepts, problem-solving steps, or any other content that is relevant to solving the problem. Return ONLY valid JSON in this exact format:
# {"related": "yes"} or {"related": "no"}"""

    system_prompt = """You are given a dialogue between a student and tutor where the student is trying to solve a math problem. Your job is to determine if the dialogue is directly related to a given math question.
A dialogue is related ONLY if it clearly refers to the same specific problem, expression, numbers, or variables as the question, or is explicitly working toward solving that exact problem.
If the dialogue discusses only a similar concept (e.g., same math skill like combining like terms) but uses different numbers, variables, or a different expression, then it is NOT related.

Return ONLY valid JSON in this exact format:
{"related": "yes"} or {"related": "no"}"""


    for _, dialogue in data.iterrows():
        question = dialogue["question"]
        dialogue_str = ""
        for turn in dialogue["turns"]:
            dialogue_str += f"{turn['role']}: {turn['content']}\n"
        prompt.append(f"Question: {question}\n\nDialogue:\n{dialogue_str}\nFollow system message and answer with 'yes' if it is related and 'no' if it is not in json format.")

    client = OpenAIClient(False)
    generation_args = {"response_format": {"type": "json_object"}}
    generation_args["temperature"] = 0.0
    generation_args["max_completion_tokens"] = 25
    set_trace()

    related_res = client.get_responses(prompt, 'gpt-4.1', system_prompt, generation_args, False)

    res, track_inv_ls = [], []
    for i in range(len(related_res)):
        try:
            rel_res_i = json.loads(related_res[i])
            res.append(rel_res_i['related'].lower())
        except:
            res.append('')
            track_inv_ls.append(i)

    print('Invalid output:', len(track_inv_ls))
    print('Related Count:', res.count('yes'))
    set_trace()

    bool_res = [r == 'yes' for r in res]

    return bool_res


###### Profile Generation through Prompting ######
def generate_profile(data):
    prompts = []
    
    for _, row in data.iterrows():
        question = row["question"]
        correct_answer = row['CorrectAnswer']
        # turn_i = row['turns']
        turn_i = [*row["turns"]] # Copy to not modify original dialogue
        correctness_i = row['correctness']
        acts_i = row['acts']

        for turn_k in correctness_i.keys():
            turn_k_ind = int(turn_k.split()[1]) - 1
            act_turn_i = acts_i[turn_k]['act']
            correctness_turn_i = correctness_i[turn_k]['correct']

            turn_i_content = turn_i[turn_k_ind]['content']
            
            # if correctness_i is None: turn_i_content_annotate contains only turn content and dialogue act, otherwise, contain turn content, dialogue act, and correctness
            if correctness_turn_i is not None:
                turn_i_content = f"{turn_i_content}     (Student dialogue act: {act_turn_i},  Correctness: {correctness_turn_i})"
            else:
                turn_i_content = f"{turn_i_content}     (Student dialogue act: {act_turn_i})"

            # turn_i[turn_k_ind]['content'] = turn_i_content
            turn_i[turn_k_ind] = {**turn_i[turn_k_ind], "content": turn_i_content}


        output_i = ''
        for idx, it in enumerate(turn_i):
            turn_idx = idx + 1
            output_i += f"\nTurn {turn_idx} ({it['role']}): {it['content']}"


        context = f"Question:\n{question}\nThe correct answer is: {correct_answer}\n\n"
        prompt = context + "Dialogue with Student Acts and Optional Correctness Labels:"

        prompt = prompt + output_i
        prompts.append(prompt.strip())


    set_trace()
    client = OpenAIClient(False)
    generation_args = {"response_format": {"type": "text"}}
    generation_args["temperature"] = 0.0
    generation_args["max_completion_tokens"] = 400

    profile_res = client.get_responses(prompts, 'gpt-4.1', PROFILE_SYSTEM_PROMPT, generation_args, False)
    set_trace()
    # print(profile_res[0])

    profile_ls = [process_result(result, "text") for result in profile_res]

    return profile_ls

###### Main ######

LABEL_TO_ANNO_FN = {
    "corr": annotate_corr,
    "ocean_personas": annotate_ocean_personas,
    "freeform_personas": annotate_freeform_personas,
    "questions": annotate_questions,
    "eedi_kcs": annotate_eedi_kcs,
    "acts": annotate_acts,
}

LABELS_WITH_TEXT_OUTPUT = {"freeform_personas"}

def annotate_split(data, prompting_fn: PromptingFnSig, args: dict):
    # data = pd.DataFrame(data_list)
    # if args["truncate"]:
    #     data = data[:args["truncate"]]
    return LABEL_TO_ANNO_FN[args["label"]](data, prompting_fn, args)


def annotate(args: dict):
    # Load data
    annotation_model = args["annotation_model"].replace("/", "-")
    # train_data, val_data = load_train_val_data(args["dataset"], annotation_model=annotation_model, drop_unsolvable=False)
    # test_data = load_test_data(args["dataset"], annotation_model=annotation_model, drop_unsolvable=False)

    data = pd.read_csv("data/annotated/eedi/student_dialogue_annotated_filt.csv")
    # data = pd.read_csv('data/annotated/eedi/student_dialogue_strict_rel.csv')
    data['turns'] = data['turns'].apply(literal_eval)


    # # try profile generation with complete annotations
    data['correctness'] = data['correctness'].apply(literal_eval)
    data['acts'] = data['acts'].apply(literal_eval)
    data['KC'] = data['KC'].apply(literal_eval)

    if args["truncate"]:
        data = data[:args["truncate"]]

    # Generate student profiles based on dialogue, baseline version use one single dialogue. (Future version try to use iterative way with previous student profile and dialogue)
    generated_profiles = generate_profile(data)
    data["gen_profile"] = generated_profiles

    set_trace()

    # Set up LLM for prompting
    response_format = "text" if args["label"] in LABELS_WITH_TEXT_OUTPUT else "json_object"
    prompting_fn = get_prompting_function({**args, "response_format": response_format})

    
    # # 1. get the Dialogue Acts label and Correctness label
    # # Conduct (dialogue, question) relatedness check as sanity check before annotation
    # res = annotate_related_dialogue_problem(data)
    # filt_val = data[res]
    # print(f"Related count: {len(filt_val)} / {len(data)}")
    # set_trace()

    
    res = annotate_split(data, prompting_fn, args)
    print(res.columns)

    # # Annotate all data splits
    # print("Annotating train split...")
    # train_data = annotate_split(train_data, prompting_fn, args)
    # print("Annotating val split...")
    # val_data = annotate_split(val_data, prompting_fn, args)
    # print("Annotating test split...")
    # test_data = annotate_split(test_data, prompting_fn, args)
    # suffix = f"_{annotation_model}"
    # if args["truncate"]:
    #     suffix += f"_truncate_{args['truncate']}"
    # save_annotated_data(train_data, val_data, test_data, args["dataset"], suffix)

def data_split():
    data = pd.read_csv("data/annotated/eedi/student_dialogue_exclude_1.csv")

    students = data["studentID"].unique()
    train_students = random.sample(list(students), int(0.8 * len(students)))
    val_students = random.sample(list(set(students) - set(train_students)), int(0.1 * len(students)))
    test_students = list(set(students) - set(train_students) - set(val_students))

    train_data = data[data["studentID"].isin(train_students)]
    val_data = data[data["studentID"].isin(val_students)]
    test_data = data[data["studentID"].isin(test_students)]

    set_trace()

    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--label", choices=["corr", "ocean_personas", "freeform_personas", "questions", "eedi_kcs", "acts"])
    parser.add_argument("--label", choices=["corr", "acts", "questions", "eedi_kcs"])
    parser.add_argument("--dataset", default="eedi")
    parser.add_argument("--truncate", type=int)
    parser.add_argument("--engine", choices=["openai", "vllm"], default="openai")
    parser.add_argument("--annotation_model", default="gpt-4.1")
    parser.add_argument("--use_azure", action="store_true")
    parser.add_argument("--batch_api", action="store_true")
    args = parser.parse_args().__dict__

    # train_data, val_data, test_data = data_split()
    annotate(args)

if __name__ == "__main__":
    main()
