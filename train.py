"""
Pipeline completo de Fine-Tuning com QLoRA
Laboratório 07 — Especialização de LLMs com LoRA e QLoRA

Passos cobertos:
  2. Configuração da quantização com bitsandbytes (4-bit NF4)
  3. Arquitetura do LoRA via peft (r=64, alpha=16, dropout=0.1)
  4. Treinamento com SFTTrainer + paged_adamw_32bit + cosine scheduler
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer

# ──────────────────────────────────────────────────────────────────────────────
# Configurações globais
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Llama-2-7b-hf"   # requer aceite dos termos na HuggingFace
OUTPUT_DIR = "./outputs/llama2-qlora-data-science"
ADAPTER_DIR = "./outputs/adapter"

TRAIN_FILE = "data/train.jsonl"
TEST_FILE = "data/test.jsonl"

MAX_SEQ_LENGTH = 512


# ──────────────────────────────────────────────────────────────────────────────
# Passo 2: Configuração da Quantização (QLoRA — 4-bit NF4)
# ──────────────────────────────────────────────────────────────────────────────
def build_bnb_config() -> BitsAndBytesConfig:
    """
    Carrega o modelo base em 4 bits usando o tipo NormalFloat 4-bit (nf4).
    O compute_dtype float16 é usado nas operações de forward/backward pass,
    enquanto os pesos ficam armazenados em 4 bits — reduzindo o uso de VRAM
    em ~4x comparado ao fp16 puro.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat 4-bit
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,      # quantização aninhada extra
    )


# ──────────────────────────────────────────────────────────────────────────────
# Passo 3: Arquitetura do LoRA
# ──────────────────────────────────────────────────────────────────────────────
def build_lora_config() -> LoraConfig:
    """
    O LoRA congela a matriz original W e aprende apenas dois fatores menores:
        W' = W + (B × A),  onde A ∈ R^{r×d}  e  B ∈ R^{d×r}
    Isso reduz o número de parâmetros treináveis de bilhões para ~milhões.

    Hiperparâmetros:
      r=64     — dimensão do espaço latente das matrizes de adaptação
      alpha=16 — fator de escala: pesos finais *= alpha / r = 0.25
      dropout  — regularização para evitar overfitting no adapter
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=[          # camadas de atenção do Llama
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Passo 4: TrainingArguments com otimizações de memória
# ──────────────────────────────────────────────────────────────────────────────
def build_training_args() -> TrainingArguments:
    """
    paged_adamw_32bit: o AdamW paginado mantém os estados do otimizador em
    memória paginada, transferindo picos de uso da GPU para a CPU quando
    necessário — essencial para GPUs com VRAM limitada.

    cosine scheduler: a taxa de aprendizado segue uma curva cosseno decrescente,
    proporcionando convergência mais suave que o decay linear.

    warmup_ratio=0.03: nos primeiros 3% dos passos, o LR sobe gradualmente de 0
    até o valor alvo, evitando gradientes explosivos no início do treino.
    """
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,        # batch efetivo = 16
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        group_by_length=True,
        report_to="none",                     # desativa W&B por padrão
    )


# ──────────────────────────────────────────────────────────────────────────────
# Formatação dos exemplos
# ──────────────────────────────────────────────────────────────────────────────
def format_prompt(example: dict) -> dict:
    """
    Formata cada exemplo no template de instrução do Llama-2 Chat:
      [INST] pergunta [/INST] resposta
    O SFTTrainer treina o modelo a prever apenas os tokens da resposta.
    """
    text = (
        f"[INST] {example['prompt'].strip()} [/INST] "
        f"{example['response'].strip()}"
    )
    return {"text": text}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # 1. Carrega datasets
    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "test": TEST_FILE},
    )
    dataset = dataset.map(format_prompt)
    print(f"Treino: {len(dataset['train'])} | Teste: {len(dataset['test'])}")

    # 2. Quantização
    bnb_config = build_bnb_config()

    # 3. Carrega modelo base e tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False             # necessário durante o treino
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. LoRA config
    lora_config = build_lora_config()

    # 5. TrainingArguments
    training_args = build_training_args()

    # 6. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    # 7. Treinamento
    print("\nIniciando fine-tuning com QLoRA...\n")
    trainer.train()

    # 8. Salva apenas os pesos do adapter (não o modelo base inteiro)
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\nAdapter salvo em: {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
