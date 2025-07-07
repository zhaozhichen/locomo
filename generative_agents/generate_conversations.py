import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import logging
import argparse
import os, json, sys
import random
from datetime import date, timedelta, datetime
from generative_agents.conversation_utils import *
from generative_agents.html_utils import convert_to_chat_html, convert_to_chat_html_multi
from generative_agents.event_utils import *
from generative_agents.memory_utils import *
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from global_methods import run_gemini, set_gemini_key
import google.generativeai as genai
import time

logging.basicConfig(level=logging.INFO)

# GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_MODEL_NAME = "gemini-2.5-pro"

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--out-dir', required=True, type=str, help="Path to directory containing agent files and downloaded images for a conversation")
    parser.add_argument('--prompt-dir', required=True, type=str, help="Path to the dirctory containing in-context examples")
    
    parser.add_argument('--start-session', type=int, default=1, help="Start iterating from this index; first session is 1")
    parser.add_argument('--num-sessions', type=int, default=20, help="Maximum number of sessions in the conversation")
    parser.add_argument('--num-days', type=int, default=240, help="Desired temporal span of the multi-session conversation")
    parser.add_argument('--num-events', type=int, default=15, help="Total number of events to generate for each agent; 1 per session works best")
    parser.add_argument('--max-turns-per-session', type=int, default=20, help="Maximum number of total turns in each session")
    parser.add_argument('--num-events-per-session', type=int, default=50, help="Total number of events to be assigned to each agent per session; 1-2 works best")

    parser.add_argument('--persona', action="store_true", help="Set flag to sample a new persona from MSC and generate details")
    parser.add_argument('--session', action="store_true", help="Set flag to generate sessions based on the generated/existing personas")
    parser.add_argument('--events', action="store_true", help="Set flag to generate and events suited to the generated/existing personas")
    parser.add_argument('--blip-caption', action="store_true", help="Set flag to use BLIP model to generate captions for downloaded images")
    parser.add_argument('--overwrite-persona', action='store_true', help="Overwrite existing persona summaries saved in the agent files")
    parser.add_argument('--overwrite-events', action='store_true', help="Overwrite existing events saved in the agent files")
    parser.add_argument('--overwrite-session', action='store_true', help="Overwrite existing sessions saved in the agent files")
    parser.add_argument('--summary', action="store_true", help="Set flag to generate and use summaries in the conversation generation prompt")

    parser.add_argument('--emb-file', type=str, default='embeddings.pkl', help="Name of the file used to save embeddings for the fine-grained retrieval-based memory module")
    parser.add_argument('--reflection', action="store_true", help="Set flag to use reflection module at the end of each session and include in the conversation generation prompt for context")

    parser.add_argument('--num-agents', type=int, default=2, help="Number of agents in the group chat")

    args = parser.parse_args()
    return args


def get_blip_caption(img_file, model, processor):

    raw_image = Image.open(img_file).convert('RGB')
    # conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt").to("cpu")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def save_agents(agents, args):
    current_names = set()
    for i, agent in enumerate(agents):
        if agent['name'] in current_names:
            raise ValueError(f"Duplicate agent name detected during save: {agent['name']}")
        current_names.add(agent['name'])
        agent_file = args.agent_file_pattern % i
        logging.info("Saving updated Agent %s to %s" % (i+1, agent_file))
        with open(agent_file, 'w') as f:
            json.dump(agent, f, indent=2)


def load_agents(args):
    agents = []
    for i in range(args.num_agents):
        agent_file = args.agent_file_pattern % i
        agent = json.load(open(agent_file))
        agents.append(agent)
    # After loading agents, store their original names
    original_names = [agent['name'] for agent in agents]
    return agents, original_names


def get_random_time():

    start_time = timedelta(hours=9, minutes=0, seconds=0)
    end_time = timedelta(hours=21, minutes=59, seconds=59)
    random_seconds = random.randint(int(start_time.total_seconds()), int(end_time.total_seconds()))
    hours = random_seconds//3600
    minutes = (random_seconds - (hours*3600))//60
    return timedelta(hours=hours, minutes=minutes, seconds=0)


def datetimeStr2Obj(dateStr):
    if 'am' in dateStr:
        datetimeObj = datetime.strptime(dateStr, "%H:%M am on %d %B, %Y")
    else:
        datetimeObj = datetime.strptime(dateStr, "%H:%M pm on %d %B, %Y")
    return datetimeObj

def datetimeObj2Str(datetimeObj):

    time_mod = 'am' if datetimeObj.hour <= 12 else 'pm'
    hour = datetimeObj.hour if datetimeObj.hour <= 12 else datetimeObj.hour-12
    min = str(datetimeObj.minute).zfill(2)
    return str(hour) + ':' + min + ' ' + time_mod + ' on ' + str(datetimeObj.day) + ' ' + datetimeObj.strftime("%B") + ', ' + str(datetimeObj.year)


def dateObj2Str(dateObj):
    return dateObj.strftime("%d") + ' ' + dateObj.strftime("%B") + ', ' + dateObj.strftime("%Y")


def get_random_date():

    # initializing dates ranges
    test_date1, test_date2 = date(2022, 1, 1), date(2023, 6, 1)
    # getting days between dates
    dates_bet = test_date2 - test_date1
    total_days = dates_bet.days
    delta_days = random.choice(range(1, total_days))
    random_date = test_date1 + timedelta(days=int(delta_days))
    return random_date


def get_session_summary(session, speaker_1, speaker_2, curr_date, previous_summary=""):
    session_query = ''
    for c in session:
        session_query += "%s: %s\n" % (c["speaker"], c["text"])
        if "image" in c:
            session_query += "[%s shares %s]\n" % (c["speaker"], c["image"])

    if previous_summary:
        query = SESSION_SUMMARY_PROMPT % (speaker_1['name'], speaker_2['name'], previous_summary, curr_date,
                                               speaker_1['name'], speaker_2['name'], session_query, speaker_1['name'], speaker_2['name'])
    else:
        query = SESSION_SUMMARY_INIT_PROMPT % (speaker_1['name'], speaker_2['name'], curr_date, session_query)

    query += '\n\n'
    # should summarize persona, previous conversations with respect to speaker.
    output = run_gemini(gemini_model, query, max_tokens=150)
    if output is None:
        logging.error("Gemini returned None for session summary. Using fallback summary.")
        output = "No summary available."
    output = output.strip()
    return output


def get_image_queries(events):

    images = [e["image"] for e in events]
    input_query = "\nInput: ".join(images)

    output = run_gemini(gemini_model, EVENT2QUERY_PROMPT % input_query, 1, 200)
    output = output.strip()
    print(output)
    json_output = clean_json_output(output)

    assert len(events) == len(json_output), [events, json_output]

    for i in range(len(events)):
        events[i]["query"] = json_output[i]
    return events


def get_all_session_summary(speaker, curr_sess_id):

    summary = "\n"
    for sess_id in range(1, curr_sess_id):
        sess_date = speaker['session_%s_date_time' % sess_id]
        sess_date = sess_date[2] + ' ' + sess_date[1] + ', ' + sess_date[0]
        summary += sess_date + ': ' + speaker["session_%s_summary" % sess_id] + '\n'
    return summary


def catch_date(date_str):
    date_format1 = '%d %B, %Y'
    date_format2 = '%d %B %Y'
    try:
        return datetime.strptime(date_str, date_format1)
    except:
        return datetime.strptime(date_str, date_format2)


def get_session_date(events, args, prev_date = None):

    agent_events = sort_events_by_time(events)
    curr_count = 0
    stop_count = args.num_events_per_session
    stop_date = None
    for e in agent_events:
        event_date =  catch_date(e['date'])
        if prev_date:
            if event_date >= prev_date:
                print("Including event %s" % json.dumps(e, indent=2))
                curr_count += 1
        else:
            print("Including event %s" % json.dumps(e, indent=2))
            curr_count += 1
        if curr_count == stop_count:
            stop_date = event_date
            break
    stop_date = event_date

    return min(stop_date, prev_date) + timedelta(days=random.choice([1, 2])) if prev_date else stop_date + timedelta(days=random.choice([1, 2]))


def get_relevant_events(events, curr_date, prev_date=None):

    events = sort_events_by_time(events)
    relevant_events = []
    for e in events:
        # event_date = datetime.strptime(e['date'], "%d %B, %Y")
        event_date = catch_date(e['date'])
        if event_date > curr_date:
            continue
        if prev_date:
            if event_date <= prev_date:
                continue
        relevant_events.append(e)

    return relevant_events


def get_event_string(session_events, all_events):

    id2events = {e['id']: e for e in all_events}

    event_string = ""
    for e in session_events:
        try:
            event_text = 'On' + e["date"] + ", " + e["sub-event"]
        except KeyError:
            event_text = 'On' + e["date"] + ", " + e["sub_event"]

        # if the event is caused by previous events, include them for context
        if len(e['caused_by']) > 0:
            event_text += ' Because previously'
            for e_id in e['caused_by']:
                try:
                    event_text += ', ' + id2events[e_id]["sub-event"] + ' (%s)' % id2events[e_id]["date"]
                except KeyError:
                    event_text += ', ' + id2events[e_id]["sub_event"] + ' (%s)' % id2events[e_id]["date"]
        
        event_string += event_text + "\n"

    return event_string


def remove_context(args, curr_dialog, prev_dialog, caption=None):

    prompt_data = json.load(open(os.path.join(args.prompt_dir, 'remove_context_examples.json')))
    if caption:
        query = prompt_data["input_format_w_image"].format(prev_dialog, curr_dialog, caption)
    else:
        query = prompt_data["input_format"].format(prev_dialog, curr_dialog)
    output = run_gemini(gemini_model, prompt_data["prompt"], 
                              [[prompt_data["input_format"].format(*example["input"]) if len(example["input"]) == 2 else prompt_data["input_format_w_image"].format(*example["input"]), example["output"]] for example in prompt_data['examples']], 
                              query, num_gen=1, num_tokens_request=128, use_16k=False)
    return output


def get_agent_query(speaker_1, all_agents, curr_sess_id=0, 
                    prev_sess_date_time='', curr_sess_date_time='', 
                    use_events=False, instruct_stop=False, dialog_id=0, last_dialog='', embeddings=None, reflection=False):

    stop_instruction = "To end the conversation, write [END] at the end of the dialog."
    if instruct_stop:
        print("**** Using stop instruction ****")

    # For multi-agent, we need to adapt the prompt to mention the group
    other_agents = [agent for agent in all_agents if agent['name'] != speaker_1['name']]
    other_names = ', '.join([agent['name'] for agent in other_agents])

    if curr_sess_id == 1:
        
        if use_events:
            events = get_event_string(speaker_1['events_session_%s' % curr_sess_id], speaker_1['graph'])
            # Adapt prompt for multi-agent scenario
            query = AGENT_CONV_PROMPT_SESS_1_W_EVENTS % (speaker_1['persona_summary'],
                    speaker_1['name'], other_names, 
                    curr_sess_date_time, speaker_1['name'],  events, speaker_1['name'], other_names, stop_instruction if instruct_stop else '')
        else:
            query = AGENT_CONV_PROMPT_SESS_1 % (speaker_1['persona_summary'],
                                speaker_1['name'], other_names, 
                                curr_sess_date_time, speaker_1['name'],  other_names, speaker_1['name'])
    
    else:
        if use_events:
            events = get_event_string(speaker_1['events_session_%s' % curr_sess_id], speaker_1['graph'])
            if dialog_id == 0:
                # if a new session is starting, get information about the topics discussed in last session
                context_from_1, context_from_2 = get_recent_context_multi(speaker_1, all_agents, curr_sess_id, reflection=reflection)
                recent_context = '\n'.join(context_from_1) + '\n' +  '\n'.join(context_from_2) # with reflection
                query = AGENT_CONV_PROMPT_W_EVENTS_V2_INIT % (speaker_1['persona_summary'],
                            speaker_1['name'], other_names, prev_sess_date_time,
                            curr_sess_date_time, speaker_1['name'],  speaker_1['session_%s_summary' % (curr_sess_id-1)], events, stop_instruction if instruct_stop else '', other_names)
                
            else:
                # during an ongoing session, get fine-grained information from a previous session using retriever modules
                past_context = get_relevant_context_multi(speaker_1, all_agents, last_dialog, embeddings, curr_sess_id, reflection=reflection)
                query = AGENT_CONV_PROMPT_W_EVENTS_V2 % (speaker_1['persona_summary'],
                            speaker_1['name'], other_names, prev_sess_date_time,
                            curr_sess_date_time, speaker_1['name'], speaker_1['session_%s_summary' % (curr_sess_id-1)], events, past_context, stop_instruction if instruct_stop else '', other_names)
        else:
            summary = get_all_session_summary(speaker_1, curr_sess_id)
            query = AGENT_CONV_PROMPT % (speaker_1['persona_summary'],
                                        speaker_1['name'], other_names, prev_sess_date_time, summary,
                                        curr_sess_date_time, speaker_1['name'],  other_names, speaker_1['name']) 
    
    return query


def get_session(agents, args, prev_date_time_string='', curr_date_time_string='', curr_sess_id=0, captioner=None, img_processor=None, reflection=False):
    import random
    # load embeddings for retrieving relevant observations from previous conversations
    if curr_sess_id == 1:
        embeddings = None
    else:
        embeddings = pkl.load(open(args.emb_file, 'rb'))

    # select one of the speakers to start the session at random
    prev_speaker_idx = None
    curr_speaker_idx = random.randint(0, len(agents)-1)
    conv_so_far = agents[curr_speaker_idx]['name'] + ': '

    session = []
    stop_dialog_count = args.max_turns_per_session if args.max_turns_per_session <= 10 else random.choice(list(range(10, args.max_turns_per_session)))
    break_flags = [False] * len(agents)
    for i in range(args.max_turns_per_session):
        if all(break_flags):
            break
        # Select next speaker (not the same as previous)
        possible_idxs = [idx for idx in range(len(agents)) if idx != prev_speaker_idx]
        curr_speaker_idx = random.choice(possible_idxs)
        curr_speaker = agents[curr_speaker_idx]
        # For now, pick a random other agent as the "listener"
        other_idxs = [idx for idx in range(len(agents)) if idx != curr_speaker_idx]
        listener_idx = random.choice(other_idxs)
        listener = agents[listener_idx]
        agent_query = get_agent_query(
            curr_speaker, agents,
            prev_sess_date_time=prev_date_time_string,
            curr_sess_date_time=curr_date_time_string,
            curr_sess_id=curr_sess_id,
            use_events=args.events,
            instruct_stop=i>=stop_dialog_count,
            dialog_id=i,
            last_dialog='' if i == 0 else session[-1]['speaker'] + ' says, ' + session[-1]['clean_text'],
            embeddings=embeddings,
            reflection=reflection
        )
        # if the speaker in previous turn sent an image, get caption + questions
        if len(session) > 1 and "img_id" in session[-1]:
            caption = "shares " + session[-1]['blip_caption']
            question = run_gemini(gemini_model, VISUAL_QUESTION_PROMPT.format(
                curr_speaker['persona_summary'],
                listener['persona_summary'],
                listener['name'], session[-1]['clean_text'], caption,
                curr_speaker['name']), max_tokens=100)
            question = question.strip() if question is not None else ""
            agent_query = agent_query + f"\nUse the following question about the photo shared by {listener['name']} in your reply: {question}."
        
        # Add retry logic for main conversation generation
        max_retries = 3
        retry_count = 0
        output = None
        
        while retry_count < max_retries and output is None:
            try:
                output = run_gemini(gemini_model, agent_query + conv_so_far, max_tokens=100)
                if output is None:
                    logging.warning(f"run_gemini returned None for conversation generation, attempt {retry_count + 1}")
            except Exception as e:
                logging.error(f"Error in conversation generation, attempt {retry_count + 1}: {e}")
                output = None
            
            if output is None:
                retry_count += 1
                if retry_count < max_retries:
                    logging.info(f"Retrying conversation generation in 2 seconds...")
                    time.sleep(2)
        
        # Fallback if all retries failed
        if output is None:
            logging.error("All retries failed for conversation generation, using fallback response")
            output = f"I'm having trouble responding right now."
        
        output = output.strip().split('\n')[0]
        output = clean_dialog(output, curr_speaker['name'])
        output = {"text": output, "raw_text": output}
        image_search_query, photo_caption = insert_image_response(output["text"], gemini_model)
        if image_search_query is not None:
            img_dir = os.path.join(args.out_dir, f'session_{curr_sess_id}', curr_speaker['name'])
            file_urls, file_names = get_images(image_search_query, img_dir, i)
            if file_names == []:
                print("Image not found, for search query: ", image_search_query)
            else:
                output["img_url"] = file_urls
                output["img_file"] = file_names
                output["img_id"] = i
                output['query'] = image_search_query
                output['caption'] = photo_caption
                if args.blip_caption:
                    output['blip_caption'] = get_blip_caption(os.path.join(img_dir, file_names[0]), captioner, img_processor).replace('photography', 'photo')
        output["speaker"] = curr_speaker["name"]
        text_replaced_caption = replace_captions(output["text"], args, gemini_model)
        if not text_replaced_caption.isspace():
            if '[END]' in output["text"]:
                output["clean_text"] = text_replaced_caption
                break_flags[curr_speaker_idx] = True
            else:
                # Add retry logic for Gemini API calls
                max_retries = 3
                retry_count = 0
                clean_text = None
                
                while retry_count < max_retries and clean_text is None:
                    try:
                        clean_text = run_gemini(gemini_model, CASUAL_DIALOG_PROMPT % text_replaced_caption, max_tokens=100)
                        if clean_text is not None:
                            clean_text = clean_text.strip()
                        else:
                            logging.warning(f"run_gemini returned None for clean_text generation, attempt {retry_count + 1}")
                    except Exception as e:
                        logging.error(f"Error in clean_text generation, attempt {retry_count + 1}: {e}")
                        clean_text = None
                    
                    if clean_text is None:
                        retry_count += 1
                        if retry_count < max_retries:
                            logging.info(f"Retrying clean_text generation in 2 seconds...")
                            time.sleep(2)
                
                # Fallback if all retries failed
                if clean_text is None:
                    logging.error("All retries failed for clean_text generation, using original text")
                    clean_text = text_replaced_caption
                
                output["clean_text"] = clean_text
        else:
            output["clean_text"] = ""
        output["dia_id"] = f'D{curr_sess_id}:{i+1}'
        session.append(output)
        print("############ ", curr_speaker['name'], ': ', output["clean_text"])
        if "caption" in output:
            print("[ {} ]".format(output["blip_caption"]))
        if "blip_caption" in output:
            conv_so_far = conv_so_far + output["clean_text"] + '[shares ' + output["blip_caption"] + ']' + '\n'
        else:
            conv_so_far = conv_so_far + output["clean_text"] + '\n'
        prev_speaker_idx = curr_speaker_idx
        
        # Add small delay to prevent rate limiting
        time.sleep(0.5)
    return session


def main():

    # get arguments
    args = parse_args()

    set_gemini_key()
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    args.emb_file = os.path.join(args.out_dir, args.emb_file)

    # create dataset directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    logging.info("Dataset directory: %s" % args.out_dir)

    args.agent_file_pattern = os.path.join(args.out_dir, 'agent_%d.json')

    
    # Step 1: Get personalities for the agents; get a randomly selected sample from the MSC dataset and expand the few-liner personas into detailed personas.
    if args.persona:
        agents = get_msc_personas(args, gemini_model, num_agents=args.num_agents)
        original_names = [agent['name'] for agent in agents]
        args.original_names = original_names
        if agents:
            save_agents(agents, args)


    # Step 2: check if events exist; if not, generate event graphs for each of the agents 
    if args.events:

        agents, original_names = load_agents(args)
        args.original_names = original_names

        if ('graph' in agents[0] and 'graph' in agents[1]) and not args.overwrite_events:
            pass
        else:
            # if 'session_1_date_time' not in agent_a:
            start_date = get_random_date() # select a random date in 2022-2023
            end_date = start_date + timedelta(days=args.num_days)
            start_date = dateObj2Str(start_date)
            end_date = dateObj2Str(end_date)
            for agent in agents:
                agent['events_start_date'] = start_date
            logging.info("Generating a random start date for the conversation")
            save_agents(agents, args)

            
            for idx, agent in enumerate(agents):
                agent_events = []
                logging.info("Generating events for Agent %s" % agent['name'])
                trials = 0
                while len(agent_events) < args.num_events:
                    logging.info("(Re)trying to generate events with dense causal connections: trial %s" % trials)
                    agent_events = get_events(agent, start_date, end_date, args)
                    agent['graph'] = agent_events
                    trials += 1
                # Save the updated agent at the correct index
                # Load existing agent file and only update new fields, preserve identity
                agent_file = args.agent_file_pattern % idx
                try:
                    with open(agent_file, 'r') as f:
                        existing_agent = json.load(f)
                    # Preserve existing identity fields
                    if 'name' in existing_agent:
                        agent['name'] = existing_agent['name']
                    if 'persona_summary' in existing_agent:
                        agent['persona_summary'] = existing_agent['persona_summary']
                    if 'msc_prompt' in existing_agent:
                        agent['msc_prompt'] = existing_agent['msc_prompt']
                except (FileNotFoundError, json.JSONDecodeError):
                    # If file doesn't exist or is corrupted, use current agent data
                    pass
                logging.info(f"Saving updated Agent {agent['name']} to {agent_file}")
                with open(agent_file, 'w') as f:
                    json.dump(agent, f, indent=2)

        # make sure keys are all lower case
        for idx, agent in enumerate(agents):
            agent_events = agent['graph']
            agent_events = [{k.lower(): v for k,v in e.items()} for e in agent_events]
            agent['graph'] = agent_events
            # Save the updated agent at the correct index
            # Load existing agent file and only update new fields, preserve identity
            agent_file = args.agent_file_pattern % idx
            try:
                with open(agent_file, 'r') as f:
                    existing_agent = json.load(f)
                # Preserve existing identity fields
                if 'name' in existing_agent:
                    agent['name'] = existing_agent['name']
                if 'persona_summary' in existing_agent:
                    agent['persona_summary'] = existing_agent['persona_summary']
                if 'msc_prompt' in existing_agent:
                    agent['msc_prompt'] = existing_agent['msc_prompt']
            except (FileNotFoundError, json.JSONDecodeError):
                # If file doesn't exist or is corrupted, use current agent data
                pass
            logging.info(f"Saving updated Agent {agent['name']} to {agent_file}")
            with open(agent_file, 'w') as f:
                json.dump(agent, f, indent=2)

    # Step 3: 
    if args.session:

        agents, original_names = load_agents(args)
        args.original_names = original_names

        if args.blip_caption: # load an image captioner
            # init_model
            img_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            captioner = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cpu")
        else:
            img_processor = None
            captioner = None

        # default start index is 1; if resuming conversation from a leter session, indicate in script arguments using --start-session
        for j in range(args.start_session, args.num_sessions+1):

            print("******************* SESSION %s ******************" % j)

            # Ensure all agents have 'graph' before proceeding (do this for all agents first)
            missing_graph = False
            for idx, agent in enumerate(agents):
                if 'graph' not in agent or not agent['graph']:
                    start_date = get_random_date()
                    end_date = start_date + timedelta(days=args.num_days)
                    start_date = dateObj2Str(start_date)
                    end_date = dateObj2Str(end_date)
                    agent['events_start_date'] = start_date
                    logging.info(f"Generating events for Agent {agent['name']} (missing 'graph')")
                    agent_events = []
                    trials = 0
                    while len(agent_events) < args.num_events:
                        logging.info(f"(Re)trying to generate events with dense causal connections: trial {trials}")
                        agent_events = get_events(agent, start_date, end_date, args)
                        agent['graph'] = agent_events
                        trials += 1
                    # Save the updated agent at the correct index
                    # Load existing agent file and only update new fields, preserve identity
                    agent_file = args.agent_file_pattern % idx
                    try:
                        with open(agent_file, 'r') as f:
                            existing_agent = json.load(f)
                        # Preserve existing identity fields
                        if 'name' in existing_agent:
                            agent['name'] = existing_agent['name']
                        if 'persona_summary' in existing_agent:
                            agent['persona_summary'] = existing_agent['persona_summary']
                        if 'msc_prompt' in existing_agent:
                            agent['msc_prompt'] = existing_agent['msc_prompt']
                    except (FileNotFoundError, json.JSONDecodeError):
                        # If file doesn't exist or is corrupted, use current agent data
                        pass
                    logging.info(f"Saving updated Agent {agent['name']} to {agent_file}")
                    with open(agent_file, 'w') as f:
                        json.dump(agent, f, indent=2)
                    missing_graph = True
            if missing_graph:
                # Reload agents from disk after all missing graphs are generated
                agents, original_names = load_agents(args)
                args.original_names = original_names
            # Now all agents in memory have 'graph' before proceeding

            # 1. Generate and assign events for all agents first
            missing_events_session = False
            for idx, agent in enumerate(agents):
                if f'events_session_{j}' not in agent or args.overwrite_session:
                    if j>1:
                        prev_date_time = datetimeStr2Obj(agent['session_%s_date_time' % (j-1)])
                        prev_date_time_string = agent['session_%s_date_time' % (j-1)]
                    else:
                        prev_date_time, prev_date_time_string = None, None
                    curr_time = get_random_time() # timedelta object
                    curr_date = get_session_date(agent['graph'], args, prev_date=prev_date_time) # datetime object
                    curr_date_time = curr_date + curr_time # datetime object
                    curr_date_time_string = datetimeObj2Str(curr_date_time)
                    relevant_events_a = get_relevant_events(agent['graph'],  curr_date_time, prev_date=prev_date_time)
                    agent[f'events_session_{j}'] = relevant_events_a
                    agent[f'session_{j}_date_time'] = curr_date_time_string
                    # Save the updated agent at the correct index
                    # Load existing agent file and only update new fields, preserve identity
                    agent_file = args.agent_file_pattern % idx
                    try:
                        with open(agent_file, 'r') as f:
                            existing_agent = json.load(f)
                        # Preserve existing identity fields
                        if 'name' in existing_agent:
                            agent['name'] = existing_agent['name']
                        if 'persona_summary' in existing_agent:
                            agent['persona_summary'] = existing_agent['persona_summary']
                        if 'msc_prompt' in existing_agent:
                            agent['msc_prompt'] = existing_agent['msc_prompt']
                    except (FileNotFoundError, json.JSONDecodeError):
                        # If file doesn't exist or is corrupted, use current agent data
                        pass
                    logging.info(f"Saving updated Agent {agent['name']} to {agent_file}")
                    with open(agent_file, 'w') as f:
                        json.dump(agent, f, indent=2)
                    missing_events_session = True
            if missing_events_session:
                # Reload agents from disk after all missing events_session are generated
                agents, original_names = load_agents(args)
                args.original_names = original_names
            # Now all agents in memory have 'events_session_{j}' before proceeding

            # 2. Generate session for all agents (session is shared)
            session_generated = False
            for idx, agent in enumerate(agents):
                if 'session_%s' % j not in agent or args.overwrite_session:
                    if not session_generated:
                        # Generate session once for all agents
                        if j>1:
                            prev_date_time = datetimeStr2Obj(agent['session_%s_date_time' % (j-1)])
                            prev_date_time_string = agent['session_%s_date_time' % (j-1)]
                            curr_date_time_string = agent['session_%s_date_time' % j]
                        else:
                            prev_date_time, prev_date_time_string = None, None
                            curr_date_time_string = agent['session_%s_date_time' % j]
                        session = get_session(agents, args,
                                              prev_date_time_string=prev_date_time_string, curr_date_time_string=curr_date_time_string, 
                                              curr_sess_id=j, captioner=captioner, img_processor=img_processor, reflection=args.reflection)
                        session_generated = True
                    
                    # Assign session to all agents
                    for agent_idx, agent_item in enumerate(agents):
                        agent_item['session_%s' % j] = session
                        # Save the updated agent at the correct index
                        # Load existing agent file and only update new fields, preserve identity
                        agent_file = args.agent_file_pattern % agent_idx
                        try:
                            with open(agent_file, 'r') as f:
                                existing_agent = json.load(f)
                            # Preserve existing identity fields
                            if 'name' in existing_agent:
                                agent_item['name'] = existing_agent['name']
                            if 'persona_summary' in existing_agent:
                                agent_item['persona_summary'] = existing_agent['persona_summary']
                            if 'msc_prompt' in existing_agent:
                                agent_item['msc_prompt'] = existing_agent['msc_prompt']
                        except (FileNotFoundError, json.JSONDecodeError):
                            # If file doesn't exist or is corrupted, use current agent data
                            pass
                        logging.info(f"Saving updated Agent {agent_item['name']} to {agent_file}")
                        with open(agent_file, 'w') as f:
                            json.dump(agent_item, f, indent=2)
                    break  # Only generate session once

            # 3. Generate facts for all agents (facts generation is shared)
            if any('session_%s_facts' % j not in agent or args.overwrite_session for agent in agents):
                facts = get_session_facts_multi(args, agents, j)
                
                # Assign facts to all agents and save
                for idx, agent in enumerate(agents):
                    agent['session_%s_facts' % j] = facts
                    print(" --------- Session %s Summary for Agent %s---------" % (j, agent['name']))
                    print(facts)
                    # Save the updated agent at the correct index
                    # Load existing agent file and only update new fields, preserve identity
                    agent_file = args.agent_file_pattern % idx
                    try:
                        with open(agent_file, 'r') as f:
                            existing_agent = json.load(f)
                        # Preserve existing identity fields
                        if 'name' in existing_agent:
                            agent['name'] = existing_agent['name']
                        if 'persona_summary' in existing_agent:
                            agent['persona_summary'] = existing_agent['persona_summary']
                        if 'msc_prompt' in existing_agent:
                            agent['msc_prompt'] = existing_agent['msc_prompt']
                    except (FileNotFoundError, json.JSONDecodeError):
                        # If file doesn't exist or is corrupted, use current agent data
                        pass
                    logging.info(f"Saving updated Agent {agent['name']} to {agent_file}")
                    with open(agent_file, 'w') as f:
                        json.dump(agent, f, indent=2)

            # 4. Generate reflections for all agents (reflection generation is shared)
            if args.reflection and any('session_%s_reflection' % j not in agent or args.overwrite_session for agent in agents):
                reflections = get_session_reflection_multi(args, agents, j)
                
                # Assign reflections to all agents and save
                for idx, agent in enumerate(agents):
                    agent_name = agent['name']
                    if agent_name in reflections:
                        agent['session_%s_reflection' % j] = reflections[agent_name]
                        print(" --------- Session %s Reflection for Agent %s---------" % (j, agent['name']))
                        print(reflections[agent_name])
                        # Save the updated agent at the correct index
                        # Load existing agent file and only update new fields, preserve identity
                        agent_file = args.agent_file_pattern % idx
                        try:
                            with open(agent_file, 'r') as f:
                                existing_agent = json.load(f)
                            # Preserve existing identity fields
                            if 'name' in existing_agent:
                                agent['name'] = existing_agent['name']
                            if 'persona_summary' in existing_agent:
                                agent['persona_summary'] = existing_agent['persona_summary']
                            if 'msc_prompt' in existing_agent:
                                agent['msc_prompt'] = existing_agent['msc_prompt']
                        except (FileNotFoundError, json.JSONDecodeError):
                            # If file doesn't exist or is corrupted, use current agent data
                            pass
                        logging.info(f"Saving updated Agent {agent['name']} to {agent_file}")
                        with open(agent_file, 'w') as f:
                            json.dump(agent, f, indent=2)

            # 5. Generate summaries for each agent individually
            for idx, agent in enumerate(agents):
                if args.summary and ('session_%s_summary' % j not in agent or args.overwrite_session):
                    summary = get_session_summary(agent['session_%s' % j], agent, agent, agent['session_%s_date_time' % j], 
                                                  previous_summary=None if j==1 else agent['session_%s_summary' % (j-1)])
                    agent['session_%s_summary' % j] = summary
                    # Save the updated agent at the correct index
                    # Load existing agent file and only update new fields, preserve identity
                    agent_file = args.agent_file_pattern % idx
                    try:
                        with open(agent_file, 'r') as f:
                            existing_agent = json.load(f)
                        # Preserve existing identity fields
                        if 'name' in existing_agent:
                            agent['name'] = existing_agent['name']
                        if 'persona_summary' in existing_agent:
                            agent['persona_summary'] = existing_agent['persona_summary']
                        if 'msc_prompt' in existing_agent:
                            agent['msc_prompt'] = existing_agent['msc_prompt']
                    except (FileNotFoundError, json.JSONDecodeError):
                        # If file doesn't exist or is corrupted, use current agent data
                        pass
                    logging.info(f"Saving updated Agent {agent['name']} to {agent_file}")
                    with open(agent_file, 'w') as f:
                        json.dump(agent, f, indent=2)

    # Generate HTML output for all agents
    convert_to_chat_html_multi(agents, outfile=os.path.join(args.out_dir, 'sessions.html'), use_events=args.events, img_dir=args.out_dir)


if __name__ == "__main__":
    main()