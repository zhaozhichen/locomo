# save generated outputs to this location
OUT_DIR=./outputs

# save embeddings to this location
EMB_DIR=./outputs

# path to LoCoMo data file
DATA_FILE_PATH=./data/locomo10.json

# filenames for different outputs
QA_OUTPUT_FILE=locomo10_qa.json
OBS_OUTPUT_FILE=locomo10_observation.json
SESS_SUMM_OUTPUT_FILE=locomo10_session_summary.json

# path to folder containing prompts and in-context examples
PROMPT_DIR=./prompt_examples

# OpenAI API Key
export OPENAI_API_KEY=

# Google API Key (read from secret file)
export GOOGLE_API_KEY=$(cat "$(dirname "$0")/../.gemini_api_key")

# Anthropic API Key
export ANTHROPIC_API_KEY=

# HuggingFace Token
export HF_TOKEN= 