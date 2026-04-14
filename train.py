"""
Pipeline completo de Fine-Tuning com QLoRA
Laboratório 07 — Especialização de LLMs com LoRA e QLoRA

Passos cobertos:
  2. Configuração da quantização com bitsandbytes (4-bit NF4)
  3. Arquitetura do LoRA via peft (r=64, alpha=16, dropout=0.1)
  4. Treinamento com SFTTrainer + paged_adamw_32bit + cosine scheduler
"""

import os
import yaml
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


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Configurações globais (lidas do config.yaml)
# ──────────────────────────────────────────────────────────────────────────────
CFG = load_config()

MODEL_ID   = CFG["model"]["id"]
OUTPUT_DIR = CFG["paths"]["output_dir"]
ADAPTER_DIR = CFG["paths"]["adapter_dir"]

TRAIN_FILE = CFG["paths"]["train_file"]
TEST_FILE = CFG["paths"]["test_file"]

MAX_SEQ_LENGTH = CFG["model"]["max_seq_length"]


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
        load_in_4bit=CFG["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=CFG["quantization"]["quant_type"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=CFG["quantization"]["double_quant"],
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
        r=CFG["lora"]["r"],
        lora_alpha=CFG["lora"]["alpha"],
        lora_dropout=CFG["lora"]["dropout"],
        bias=CFG["lora"]["bias"],
        target_modules=CFG["lora"]["target_modules"],
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
        num_train_epochs=CFG["training"]["epochs"],
        per_device_train_batch_size=CFG["training"]["batch_size"],
        gradient_accumulation_steps=CFG["training"]["gradient_accumulation_steps"],
        optim=CFG["training"]["optimizer"],
        learning_rate=CFG["training"]["learning_rate"],
        lr_scheduler_type=CFG["training"]["lr_scheduler"],
        warmup_ratio=CFG["training"]["warmup_ratio"],
        logging_steps=CFG["training"]["logging_steps"],
        save_strategy=CFG["training"]["save_strategy"],
        evaluation_strategy=CFG["training"]["save_strategy"],
        fp16=CFG["training"]["fp16"],
        bf16=False,
        max_grad_norm=CFG["training"]["max_grad_norm"],
        group_by_length=CFG["training"]["group_by_length"],
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
