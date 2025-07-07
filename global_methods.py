import openai
import numpy as np
import json
import time
import sys
import os
import re
import random

import google.generativeai as genai
from anthropic import Anthropic

GEMINI_MODEL_NAME = "gemini-2.5-pro"

def get_openai_embedding(texts, model="text-embedding-ada-002"):
   texts = [text.replace("\n", " ") for text in texts]
   return np.array([openai.Embedding.create(input = texts, model=model)['data'][i]['embedding'] for i in range(len(texts))])

def set_anthropic_key():
    pass

def set_gemini_key():

    # Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def set_openai_key():
    openai.api_key = os.environ['OPENAI_API_KEY']


def run_json_trials(query, num_gen=1, num_tokens_request=1000, 
                model='davinci', use_16k=False, temperature=1.0, wait_time=1, examples=None, input=None):

    run_loop = True
    counter = 0
    while run_loop:
        try:
            if examples is not None and input is not None:
                # Use Gemini for few-shot
                from global_methods import set_gemini_key
                import google.generativeai as genai
                set_gemini_key()
                gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                output = run_gemini_with_examples(gemini_model, query, examples, input, max_tokens=num_tokens_request)
                print("Raw output from Gemini before JSON parsing:")
                print(repr(output))
                output = output.strip() if output else ""
            else:
                output = run_chatgpt(query, num_gen=num_gen, wait_time=wait_time, model=model,
                                                   num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature)
                print("Raw output from LLM before JSON parsing:")
                print(repr(output))
            output = output.replace('json', '') # this frequently happens
            output_clean = re.sub(r"^```(?:json)?\\n|```$", "", output.strip(), flags=re.MULTILINE)
            facts = json.loads(output_clean.strip())
            run_loop = False
        except json.decoder.JSONDecodeError:
            counter += 1
            time.sleep(1)
            print("Retrying to avoid JsonDecodeError, trial %s ..." % counter)
            print("Failed output:", repr(output))
            if counter == 10:
                print("Exiting after 10 trials")
                sys.exit()
            continue
    return facts


def run_claude(query, max_new_tokens, model_name):

    if model_name == 'claude-sonnet':
        model_name = "claude-3-sonnet-20240229"
    elif model_name == 'claude-haiku':
        model_name = "claude-3-haiku-20240307"

    client = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    # print(query)
    message = client.messages.create(
        max_tokens=max_new_tokens,
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model=model_name,
    )
    print(message.content)
    return message.content[0].text


def run_gemini(model, content: str, max_tokens: int = 0, max_retries: int = 3, base_wait_time: float = 1.0):
    """
    Run Gemini with robust error handling and retry logic.
    
    Args:
        model: Gemini model instance
        content: Input text to generate response for
        max_tokens: Maximum tokens (not used by Gemini but kept for compatibility)
        max_retries: Maximum number of retry attempts
        base_wait_time: Base wait time for exponential backoff
    
    Returns:
        Generated text or None if all retries failed
    """
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(content)
            if response and hasattr(response, 'text'):
                return response.text
            else:
                print(f"Gemini returned empty response on attempt {attempt + 1}")
                return None
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Check for specific error types that should trigger retries
            should_retry = any(keyword in error_msg.lower() for keyword in [
                'deadline exceeded', '504', 'timeout', 'rate limit', 'quota', 
                'resource exhausted', 'unavailable', 'internal error'
            ])
            
            if attempt < max_retries and should_retry:
                # Exponential backoff with jitter
                wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 1)
                print(f'Gemini API error on attempt {attempt + 1}/{max_retries + 1}: {error_type}: {error_msg}')
                print(f'Retrying in {wait_time:.2f} seconds...')
                time.sleep(wait_time)
            else:
                print(f'Gemini API error (final attempt): {error_type}: {error_msg}')
                return None
    
    return None


def run_chatgpt(query, num_gen=1, num_tokens_request=1000, 
                model='chatgpt', use_16k=False, temperature=1.0, wait_time=1):

    completion = None
    while completion is None:
        wait_time = wait_time * 2
        try:
            # if model == 'davinci':
            #     completion = openai.Completion.create(
            #                     # model = "gpt-3.5-turbo",
            #                     model = "text-davinci-003",
            #                     temperature = temperature,
            #                     max_tokens = num_tokens_request,
            #                     n=num_gen,
            #                     prompt=query
            #                 )
            if model == 'chatgpt':
                messages = [
                        {"role": "system", "content": query}
                    ]
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature = temperature,
                    max_tokens = num_tokens_request,
                    n=num_gen,
                    messages = messages
                )
            elif 'gpt-4' in model:
                completion = openai.ChatCompletion.create(
                    model=model,
                    temperature = temperature,
                    max_tokens = num_tokens_request,
                    n=num_gen,
                    messages = [
                        {"role": "user", "content": query}
                    ]
                )
            else:
                print("Did not find model %s" % model)
                raise ValueError
        except openai.error.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        except openai.error.ServiceUnavailableError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        # except Exception as e:
        #     if e:
        #         print(e)
        #         print(f"Timeout error, retrying after waiting for {wait_time} seconds")
        #         time.sleep(wait_time)
    

    if model == 'davinci':
        outputs = [choice.get('text').strip() for choice in completion.get('choices')]
        if num_gen > 1:
            return outputs
        else:
            # print(outputs[0])
            return outputs[0]
    else:
        # print(completion.choices[0].message.content)
        return completion.choices[0].message.content
    

def run_chatgpt_with_examples(query, examples, input, num_gen=1, num_tokens_request=1000, use_16k=False, wait_time = 1, temperature=1.0):

    completion = None
    
    messages = [
        {"role": "system", "content": query}
    ]
    for inp, out in examples:
        messages.append(
            {"role": "user", "content": inp}
        )
        messages.append(
            {"role": "system", "content": out}
        )
    messages.append(
        {"role": "user", "content": input}
    )   
    
    while completion is None:
        wait_time = wait_time * 2
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo" if not use_16k else "gpt-3.5-turbo-16k",
                temperature = temperature,
                max_tokens = num_tokens_request,
                n=num_gen,
                messages = messages
            )
        except openai.error.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        except openai.error.ServiceUnavailableError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
    
    return completion.choices[0].message.content

def run_gemini_with_examples(model, query, examples, input, max_tokens=512, max_retries=3):
    """
    Mimics run_chatgpt_with_examples for Gemini with robust error handling.
    - model: Gemini model instance
    - query: system prompt or instruction
    - examples: list of (input, output) tuples for few-shot learning
    - input: the new input to generate a response for
    - max_tokens: (optional) max tokens for the response
    - max_retries: maximum number of retry attempts
    """
    prompt = query.strip() + "\n\n"
    for inp, out in examples:
        prompt += f"Input: {inp}\nOutput: {out}\n\n"
    prompt += f"Input: {input}\nOutput:"
    
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(prompt)
            if response and hasattr(response, 'text'):
                return response.text
            else:
                print(f"Gemini returned empty response on attempt {attempt + 1}")
                return None
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Check for specific error types that should trigger retries
            should_retry = any(keyword in error_msg.lower() for keyword in [
                'deadline exceeded', '504', 'timeout', 'rate limit', 'quota', 
                'resource exhausted', 'unavailable', 'internal error'
            ])
            
            if attempt < max_retries and should_retry:
                # Exponential backoff with jitter
                wait_time = 1.0 * (2 ** attempt) + random.uniform(0, 1)
                print(f'Gemini API error on attempt {attempt + 1}/{max_retries + 1}: {error_type}: {error_msg}')
                print(f'Retrying in {wait_time:.2f} seconds...')
                time.sleep(wait_time)
            else:
                print(f'Gemini API error (final attempt): {error_type}: {error_msg}')
                return None
    
    return None
