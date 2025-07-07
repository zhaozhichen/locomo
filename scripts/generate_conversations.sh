source scripts/env.sh

python3 generative_agents/generate_conversations.py \
    --out-dir ./data/multimodal_dialog/group_example_7/ \
    --prompt-dir ./prompt_examples \
    --events --session --summary \
    --persona --blip-caption \
    --num-agents 5
