import os, json
import time
import openai
import logging
from datetime import datetime
from global_methods import run_gemini, set_gemini_key
import tiktoken
import google.generativeai as genai
import re
logging.basicConfig(level=logging.INFO)

GEMINI_MODEL_NAME = "gemini-2.5-pro"

EVENT_KG_FROM_PERSONA_PROMPT_SEQUENTIAL_INIT = """
Let's write a graph representing sub-events that occur in a person's life based on a short summary of their personality. Nodes represent sub-events and edges represent the influence of past sub-events on a current sub-event.
- The graph is represented in the form of a json list. 
- Each entry is a dictionary containing the following keys: "sub-event", "time", "caused_by", "id". 
- The "sub-event" field contains a short description of the sub-event. 
- The "time" field contains a date.
- The "id" field contains a unique identifier for the sub-event.
- The "caused_by" field represents edges and is a list of "id" of existing sub-events that have caused this sub-event. Sub-events in the "caused_by" field should occur on dates before the sub-event they have caused. Generate as many causal connections as possible.
- An example of a causal effect is when the sub-event "started a vegetable garden" causes "harvested tomatoes".
- Sub-events can be positive or negative life events.

For example,

PERSONALITY: %s
OUTPUT: %s

----------------------------------------------------------------------------------------------------------------

For the following input personality, generate three independent sub-events E1, E2 and E3 aligned with their personality. Sub-events can be positive or negative life events and should reflect evolution in the person's relationships, state of mind, personality etc. 

PERSONALITY: %s
OUTPUT: 
"""



EVENT_KG_FROM_PERSONA_PROMPT_SEQUENTIAL_CONTINUE = """
Let's write a graph representing sub-events that occur in a person's life based on a short summary of their personality. Nodes represent sub-events and edges represent the influence of past sub-events on a current sub-event.
- The graph is represented in the form of a json list. 
- Each entry is a dictionary containing the following keys: "sub-event", "time", "caused_by", "id". 
- The "sub-event" field contains a short description of the sub-event. 
- The "time" field contains a date.
- The "id" field contains a unique identifier for the sub-event.
- The "caused_by" field represents edges and is a list of "id" of existing sub-events that have caused this sub-event. Sub-events in the "caused_by" field should occur on dates before the sub-event they have caused. Generate as many causal connections as possible.
- An example of a causal effect is when the sub-event "started a vegetable garden" causes "harvested tomatoes".
- Sub-events can be positive or negative life events.
- Do not generate outdoor activities as sub-events.

For example,

PERSONALITY: %s
OUTPUT: %s

----------------------------------------------------------------------------------------------------------------

For the following input personality, generate new sub-events %s that are caused by one or more EXISTING sub-events. Sub-events can be positive or negative life events and should reflect evolution in the person's relationships, state of mind, personality etc. Do not repeat existing sub-events. Start and end your answer with a square bracket.

PERSONALITY: %s
EXISTING: %s
OUTPUT:  
"""

def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding_name = 'cl100k_base' if model_name in ['gpt-4', 'gpt-3.5-turbo', 'text-embedding-ada-002'] else 'p50k_base'
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def sort_events_by_time(graph):

    def catch_date(date_str):
        date_format1 = '%d %B, %Y'
        date_format2 = '%d %B %Y'
        try:
            return datetime.strptime(date_str, date_format1)
        except:
            return datetime.strptime(date_str, date_format2)
    
    dates = [catch_date(node['date']) for node in graph]
    sorted_dates = sorted(enumerate(dates), key=lambda t: t[1])
    graph = [graph[idx] for idx, _ in sorted_dates]
    return graph

def strip_code_block_markers(text):
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text, flags=re.IGNORECASE)
    return text.strip()

# get events in one initialization step and one or more continuation steps.
def get_events(agent, start_date, end_date, args):
    max_retries = 10
    task = json.load(open(os.path.join(args.prompt_dir, 'event_generation_examples.json')))
    persona_examples = [e["input"] + '\nGenerate events between 1 January, 2020 and 30 April, 2020.' for e in task['examples']]
    task = json.load(open(os.path.join(args.prompt_dir, 'graph_generation_examples.json')))
    input = agent['persona_summary'] + '\nAssign dates between %s and %s.' % (start_date, end_date)
    query = EVENT_KG_FROM_PERSONA_PROMPT_SEQUENTIAL_INIT % (persona_examples[0], 
                                                                   json.dumps(task['examples'][0]["output"][:12], indent=2),
                                                                   input)
    logging.info("Generating initial events")
    for attempt in range(max_retries):
        try:
            output = run_gemini(gemini_model, query, max_tokens=512)
            if not output or not output.strip():
                raise ValueError("Gemini returned empty output")
            output_clean = strip_code_block_markers(output)
            agent_events = json.loads(output_clean)
            break
        except Exception as e:
            logging.error(f"Error in get_events (init), attempt {attempt+1}: {e}\nOutput: {repr(output) if 'output' in locals() else 'None'}")
            time.sleep(1)
            if attempt == max_retries - 1:
                raise
    print("The following events have been generated in the initialization step:")
    for e in agent_events:
        print(list(e.items()))
    # Step 2: continue generation
    while len(agent_events) < args.num_events:
        logging.info("Generating next set of events; current tally = %s" % len(agent_events))
        last_event_id = agent_events[-1]["id"]
        next_event_ids = ['E' + str(i) for i in list(range(int(last_event_id[1:]) + 1, int(last_event_id[1:]) + 5))]
        next_event_id_string = ', '.join(next_event_ids[:3]) + ' and ' + next_event_ids[-1] 
        query = EVENT_KG_FROM_PERSONA_PROMPT_SEQUENTIAL_CONTINUE % (persona_examples[0], 
                                                                   json.dumps(task['examples'][0]["output"][:12], indent=2),
                                                                   next_event_id_string,
                                                                   input,
                                                                   json.dumps(agent_events, indent=2)
                                                                   )
        query_length = num_tokens_from_string(query, 'gpt-3.5-turbo')
        request_length = min(1024, 4096-query_length)
        for attempt in range(max_retries):
            try:
                output = run_gemini(gemini_model, query, max_tokens=request_length)
                if not output or not output.strip():
                    raise ValueError("Gemini returned empty output")
                output_clean = strip_code_block_markers(output)
                new_events = json.loads(output_clean)
                break
            except Exception as e:
                logging.error(f"Error in get_events (continue), attempt {attempt+1}: {e}\nOutput: {repr(output) if 'output' in locals() else 'None'}")
                time.sleep(1)
                if attempt == max_retries - 1:
                    raise
        existing_eids = [e["id"] for e in agent_events]
        agent_events.extend([o for o in new_events if o["id"] not in existing_eids])
        print("Adding events:")
        for e in agent_events:
            print(list(e.items()))
        # filter out standalone events
        if len(agent_events) > args.num_events:
            agent_events = filter_events(agent_events)
    return agent_events


def filter_events(events):

    id2events = {e["id"]: e for e in events}
    remove_ids = []
    for id in id2events.keys():
        # print(id)
        has_child = False
        # check if event has parent
        if len(id2events[id]["caused_by"]) > 0:
            continue
        # check if event has children
        for e in events:
            if id in e["caused_by"]:
                # print("Found %s in %s" % (id, e['id']))
                has_child = True
        
        if not has_child:
            # print("Did not find any connections for %s" % id)
            remove_ids.append(id)
    
    print("*** Removing %s standalone events from %s events: %s ***" % (len(remove_ids), len(id2events), ', '.join(remove_ids)))
    # for id in remove_ids:
        # print(id2events[id])
    
    return [e for e in events if e["id"] not in remove_ids]

# Initialize Gemini model at the start of the file or in main logic:
set_gemini_key()
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
