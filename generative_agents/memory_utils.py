import os, json
import time
import openai
import logging
from datetime import datetime
from global_methods import run_json_trials
import numpy as np
import pickle as pkl
import random
import google.generativeai as genai
logging.basicConfig(level=logging.INFO)


REFLECTION_INIT_PROMPT = "{}\n\nGiven the information above, what are the three most salient insights that {} has about {}? Give concise answers in the form of a json list where each entry is a string."

REFLECTION_CONTINUE_PROMPT = "{} has the following insights about {} from previous interactions.{}\n\nTheir next conversation is as follows:\n\n{}\n\nGiven the information above, what are the three most salient insights that {} has about {} now? Give concise answers in the form of a json list where each entry is a string."

SELF_REFLECTION_INIT_PROMPT = "{}\n\nGiven the information above, what are the three most salient insights that {} has about self? Give concise answers in the form of a json list where each entry is a string."

SELF_REFLECTION_CONTINUE_PROMPT = "{} has the following insights about self.{}\n\n{}\n\nGiven the information above, what are the three most salient insights that {} has about self now? Give concise answers in the form of a json list where each entry is a string."

# Multi-agent reflection prompts
MULTI_AGENT_REFLECTION_INIT_PROMPT = "{}\n\nGiven the group conversation above, what are the three most salient insights that {} has about {}? Give concise answers in the form of a json list where each entry is a string."

MULTI_AGENT_REFLECTION_CONTINUE_PROMPT = "{} has the following insights about {} from previous interactions.{}\n\nTheir next group conversation is as follows:\n\n{}\n\nGiven the information above, what are the three most salient insights that {} has about {} now? Give concise answers in the form of a json list where each entry is a string."

MULTI_AGENT_SELF_REFLECTION_INIT_PROMPT = "{}\n\nGiven the group conversation above, what are the three most salient insights that {} has about self? Give concise answers in the form of a json list where each entry is a string."

MULTI_AGENT_SELF_REFLECTION_CONTINUE_PROMPT = "{} has the following insights about self.{}\n\n{}\n\nGiven the information above, what are the three most salient insights that {} has about self now? Give concise answers in the form of a json list where each entry is a string."


CONVERSATION2FACTS_PROMPT = """
Write a concise and short list of all possible OBSERVATIONS about each speaker that can be gathered from the CONVERSATION. Each dialog in the conversation contains a dialogue id within square brackets. Each observation should contain a piece of information about the speaker, and also include the dialog id of the dialogs from which the information is taken. The OBSERVATIONS should be objective factual information about the speaker that can be used as a database about them. Avoid abstract observations about the dynamics between the speakers such as 'speaker is supportive', 'speaker appreciates' etc. Do not leave out any information from the CONVERSATION. Important: Escape all double-quote characters within string output with backslash.\n\n
"""


RETRIEVAL_MODEL = "text-embedding-ada-002" # contriever dragon dpr


def get_embedding(texts, model="models/embedding-001", task_type="retrieval_document"):
    if isinstance(texts, str):
        texts = [texts]
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=model,
            content=text,
            task_type=task_type
        )
        embeddings.append(result['embedding'])
    return np.array(embeddings)


# NEW MULTI-AGENT FUNCTIONS

def get_session_facts_multi(args, agents, session_idx, return_embeddings=True):
    """
    Multi-agent version of get_session_facts that handles N agents natively.
    
    Args:
        args: Arguments containing prompt directory and embedding file path
        agents: List of agent dictionaries
        session_idx: Session index to process
        return_embeddings: Whether to generate and save embeddings
        
    Returns:
        Dictionary mapping agent names to their facts
    """
    # Step 1: get events from any agent (they all share the same session)
    task = json.load(open(os.path.join(args.prompt_dir, 'fact_generation_examples_new.json')))
    query = CONVERSATION2FACTS_PROMPT
    examples = [[task['input_prefix'] + e["input"], json.dumps(e["output"], indent=2)] for e in task['examples']]

    # Use the first agent's session data (all agents share the same session)
    main_agent = agents[0]
    conversation = ""
    conversation += main_agent['session_%s_date_time' % session_idx] + '\n'
    
    for i, dialog in enumerate(main_agent['session_%s' % session_idx]):
        try:
            conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"'
        except KeyError:
            conversation += "[%s] " % dialog["dia_id"] + dialog['speaker'] + ' said, \"' + dialog['text'] + '\"'

        if 'blip_caption' in dialog:
            conversation += ' and shared ' + dialog['blip_caption']
        conversation += '\n'
    
    input_text = task['input_prefix'] + conversation
    facts = run_json_trials(query, num_gen=1, num_tokens_request=500, use_16k=False, examples=examples, input=input_text)

    if not return_embeddings:
        return facts

    # Ensure all agents have entries in facts dictionary
    for agent in agents:
        if agent['name'] not in facts:
            logging.warning(f"Agent {agent['name']} not found in facts. Available keys: {list(facts.keys())}")
            facts[agent['name']] = []

    # Generate embeddings for all agents
    agent_embeddings = {}
    for agent in agents:
        agent_name = agent['name']
        if facts[agent_name]:
            agent_embeddings[agent_name] = get_embedding([
                main_agent['session_%s_date_time' % session_idx] + ', ' + fact[0] 
                for fact in facts[agent_name]
            ])
        else:
            agent_embeddings[agent_name] = np.array([]).reshape(0, 768)  # Empty embedding array

    # Load or create embeddings file
    if session_idx > 1:
        try:
            with open(args.emb_file, 'rb') as f:
                embs = pkl.load(f)
        except FileNotFoundError:
            embs = {}
    else:
        embs = {}

    # Update embeddings for all agents
    for agent_name, embeddings in agent_embeddings.items():
        if embeddings.size > 0:
            if agent_name in embs:
                embs[agent_name] = np.concatenate([embs[agent_name], embeddings], axis=0)
            else:
                embs[agent_name] = embeddings

    # Save embeddings
    with open(args.emb_file, 'wb') as f:
        pkl.dump(embs, f)
    
    return facts


def get_session_reflection_multi(args, agents, session_idx):
    """
    Multi-agent version of get_session_reflection that handles N agents natively.
    
    Args:
        args: Arguments containing configuration
        agents: List of agent dictionaries
        session_idx: Session index to process
        
    Returns:
        Dictionary mapping agent names to their reflection data
    """
    # Use the first agent's session data (all agents share the same session)
    main_agent = agents[0]
    
    # Step 1: get conversation
    conversation = ""
    conversation += main_agent['session_%s_date_time' % session_idx] + '\n'
    for dialog in main_agent['session_%s' % session_idx]:
        conversation += dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"\n'

    reflections = {}
    
    # Generate reflections for each agent
    for agent in agents:
        agent_name = agent['name']
        
        # Step 2: Self-reflections
        if session_idx == 1:
            agent_self = run_json_trials(
                MULTI_AGENT_SELF_REFLECTION_INIT_PROMPT.format(conversation, agent_name), 
                model='chatgpt', num_tokens_request=300
            )
        else:
            prev_self_reflections = agent.get('session_%s_reflection' % (session_idx-1), {}).get('self', [])
            agent_self = run_json_trials(
                MULTI_AGENT_SELF_REFLECTION_CONTINUE_PROMPT.format(
                    agent_name, '\n'.join(prev_self_reflections), conversation, agent_name
                ), 
                model='chatgpt', num_tokens_request=300
            )

        # Step 3: Reflections about other agents
        other_reflections = {}
        for other_agent in agents:
            if other_agent['name'] != agent_name:
                other_name = other_agent['name']
                
                if session_idx == 1:
                    agent_on_other = run_json_trials(
                        MULTI_AGENT_REFLECTION_INIT_PROMPT.format(conversation, agent_name, other_name), 
                        model='chatgpt', num_tokens_request=300
                    )
                else:
                    prev_other_reflections = agent.get('session_%s_reflection' % (session_idx-1), {}).get('others', {}).get(other_name, [])
                    agent_on_other = run_json_trials(
                        MULTI_AGENT_REFLECTION_CONTINUE_PROMPT.format(
                            agent_name, other_name, '\n'.join(prev_other_reflections), 
                            conversation, agent_name, other_name
                        ), 
                        model='chatgpt', num_tokens_request=300
                    )
                
                # Ensure we have a list
                if isinstance(agent_on_other, dict):
                    agent_on_other = list(agent_on_other.values())
                
                other_reflections[other_name] = agent_on_other

        # Ensure we have a list for self-reflection
        if isinstance(agent_self, dict):
            agent_self = list(agent_self.values())

        reflections[agent_name] = {
            'self': agent_self,
            'others': other_reflections
        }

    return reflections


def get_recent_context_multi(agent, agents, sess_id, context_length=2, reflection=False):
    """
    Multi-agent version of get_recent_context.
    
    Args:
        agent: The agent for whom to get context
        agents: List of all agents
        sess_id: Current session ID
        context_length: Number of recent facts to retrieve
        reflection: Whether to include reflection data
        
    Returns:
        Tuple of (own_facts, other_facts) where other_facts includes all other agents
    """
    agent_name = agent['name']
    
    # Get facts about this agent
    own_facts = []
    for i in range(1, sess_id):
        session_facts = agent.get('session_%s_facts' % i, {})
        if agent_name in session_facts:
            own_facts += [
                agent['session_%s_date_time' % i] + ': ' + (f if isinstance(f, str) else ' '.join(map(str, f)))
                for f in session_facts[agent_name]
            ]
    
    # Get facts about other agents
    other_facts = []
    for i in range(1, sess_id):
        session_facts = agent.get('session_%s_facts' % i, {})
        for other_agent in agents:
            if other_agent['name'] != agent_name and other_agent['name'] in session_facts:
                other_facts += [
                    agent['session_%s_date_time' % i] + ': ' + (f if isinstance(f, str) else ' '.join(map(str, f)))
                    for f in session_facts[other_agent['name']]
                ]
    
    if reflection and sess_id > 1:
        prev_reflection = agent.get('session_%s_reflection' % (sess_id-1), {})
        self_reflections = prev_reflection.get('self', [])
        
        # Combine reflections about all other agents
        other_reflections = []
        for other_name, reflections_list in prev_reflection.get('others', {}).items():
            other_reflections.extend(reflections_list)
        
        return (own_facts[-context_length:] + self_reflections, 
                other_facts[-context_length:] + other_reflections)
    else:
        return own_facts[-context_length:], other_facts[-context_length:]


def get_relevant_context_multi(agent, agents, input_dialogue, embeddings, sess_id, context_length=2, reflection=False):
    """
    Multi-agent version of get_relevant_context.
    
    Args:
        agent: The agent for whom to get context
        agents: List of all agents
        input_dialogue: The dialogue to find relevant context for
        embeddings: Embedding dictionary
        sess_id: Current session ID
        context_length: Number of relevant facts to retrieve
        reflection: Whether to include reflection data
        
    Returns:
        Tuple of (own_context, other_context)
    """
    logging.info("Getting relevant context for response to %s (session %s)" % (input_dialogue, sess_id))
    
    own_context, other_context = get_recent_context_multi(agent, agents, sess_id, 10000, reflection=False)
    input_embedding = get_embedding([input_dialogue])
    
    agent_name = agent['name']
    
    # Check for missing embeddings
    if agent_name not in embeddings:
        logging.warning(f"Missing embeddings for agent {agent_name}. Returning empty context.")
        return [], []
    
    # Get similarities for own context
    if own_context:
        own_sims = np.dot(embeddings[agent_name], input_embedding[0])
        max_own_context = min(context_length, len(own_context))
        top_own_sims = np.argsort(own_sims)[::-1][:max_own_context]
        top_own_sims = [idx for idx in top_own_sims if idx < len(own_context)]
        relevant_own_context = [own_context[idx] for idx in top_own_sims]
    else:
        relevant_own_context = []
    
    # Get similarities for other agents' context
    relevant_other_context = []
    for other_agent in agents:
        if other_agent['name'] != agent_name and other_agent['name'] in embeddings:
            other_contexts, _ = get_recent_context_multi(other_agent, agents, sess_id, 10000, reflection=False)
            if other_contexts:
                other_sims = np.dot(embeddings[other_agent['name']], input_embedding[0])
                max_other_context = min(context_length//len(agents), len(other_contexts))
                top_other_sims = np.argsort(other_sims)[::-1][:max_other_context]
                top_other_sims = [idx for idx in top_other_sims if idx < len(other_contexts)]
                relevant_other_context.extend([other_contexts[idx] for idx in top_other_sims])
    
    if reflection and sess_id > 1:
        prev_reflection = agent.get('session_%s_reflection' % (sess_id-1), {})
        self_reflections = prev_reflection.get('self', [])
        
        # Sample reflections about others
        other_reflections = []
        for other_name, reflections_list in prev_reflection.get('others', {}).items():
            sample_size = min(context_length//4, len(reflections_list))
            if sample_size > 0:
                other_reflections.extend(random.sample(reflections_list, k=sample_size))
        
        reflection_sample_size = min(context_length//2, len(self_reflections))
        if reflection_sample_size > 0:
            relevant_own_context.extend(random.sample(self_reflections, k=reflection_sample_size))
        
        relevant_other_context.extend(other_reflections)
    
    return relevant_own_context, relevant_other_context


# LEGACY FUNCTIONS (for backward compatibility)

def get_session_facts(args, agent_a, agent_b, session_idx, return_embeddings=True):
    """Legacy function - redirects to multi-agent version"""
    agents = [agent_a]
    if agent_b['name'] != agent_a['name']:
        agents.append(agent_b)
    
    return get_session_facts_multi(args, agents, session_idx, return_embeddings)


def get_session_reflection(args, agent_a, agent_b, session_idx):
    """Legacy function - redirects to multi-agent version"""
    agents = [agent_a]
    if agent_b['name'] != agent_a['name']:
        agents.append(agent_b)
    
    multi_reflections = get_session_reflection_multi(args, agents, session_idx)
    
    # Convert back to legacy format
    reflections = {}
    if agent_a['name'] in multi_reflections:
        reflections['a'] = {
            'self': multi_reflections[agent_a['name']]['self'],
            'other': multi_reflections[agent_a['name']]['others'].get(agent_b['name'], [])
        }
    
    if agent_b['name'] in multi_reflections and agent_b['name'] != agent_a['name']:
        reflections['b'] = {
            'self': multi_reflections[agent_b['name']]['self'],
            'other': multi_reflections[agent_b['name']]['others'].get(agent_a['name'], [])
        }
    
    return reflections


def get_recent_context(agent_a, agent_b, sess_id, context_length=2, reflection=False):
    """Legacy function - redirects to multi-agent version"""
    agents = [agent_a]
    if agent_b['name'] != agent_a['name']:
        agents.append(agent_b)
    
    return get_recent_context_multi(agent_a, agents, sess_id, context_length, reflection)


def get_relevant_context(agent_a, agent_b, input_dialogue, embeddings, sess_id, context_length=2, reflection=False):
    """Legacy function - redirects to multi-agent version"""
    agents = [agent_a]
    if agent_b['name'] != agent_a['name']:
        agents.append(agent_b)
    
    return get_relevant_context_multi(agent_a, agents, input_dialogue, embeddings, sess_id, context_length, reflection)

