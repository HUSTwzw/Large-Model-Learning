#MODEL_NAME="huggyllama/llama-7b"
MODEL_NAME="NousResearch/Llama-2-7b-hf"
MODEL_PATH="model"
TOKENIZER_PATH="tokenizer"


DATA_NAME="EdinburghNLP/xsum"
DATA_PATH="dataset"


MAX_LEN=1024


BATCH_SIZE=8


R=8     #LoRA秩的大小:秩越小越节省参数
LORA_ALPHA=32       #缩放系数:越大表示LoRA层对模型参数影响越大
LORA_DROPOUT=0.05       #LoRA的dropout
TARGET_MODULES=["q_proj","v_proj"]      #确定应该对模型哪一部分参数进行LoRA(通常对Q和V进行LoRA,经过实验发现K进行LoRA的效果不明显)


LR=2e-4
NUM_EPOCHS=3


LORA_PATH="final_lora"
NEW_TOKENIZER_PATH="final_tokenizer"


OUTPUT_MAX_LEN=64