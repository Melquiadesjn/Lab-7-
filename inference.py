"""
Inferência com o modelo fine-tunado

Carrega o modelo base + adapter LoRA e gera respostas para
novas perguntas, permitindo avaliar qualitativamente o resultado
do fine-tuning antes de fazer o merge completo.

Uso:
    python inference.py
    python inference.py --prompt "Como ler um CSV com pandas?"
"""

import argparse
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["quant_type"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=cfg["quantization"]["double_quant"],
    )


def load_model_and_tokenizer(cfg: dict):
    """Carrega modelo base quantizado e aplica o adapter LoRA salvo."""
    bnb_config = build_bnb_config(cfg)

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["id"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Sobrepõe o adapter treinado ao modelo base
    model = PeftModel.from_pretrained(base_model, cfg["paths"]["adapter_dir"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["id"], trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Formata o prompt no template Llama-2 e gera a resposta."""
    formatted = f"[INST] {prompt.strip()} [/INST]"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Remove os tokens do prompt da saída
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


SAMPLE_PROMPTS = [
    "Como remover linhas com valores nulos de um DataFrame pandas?",
    "Qual a diferença entre loc e iloc no pandas?",
    "Como normalizar features com MinMaxScaler do scikit-learn?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferência com o modelo fine-tunado")
    parser.add_argument("--prompt", type=str, default=None, help="Pergunta para o modelo")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    cfg = load_config()
    print("Carregando modelo e adapter... (pode demorar alguns minutos)")
    model, tokenizer = load_model_and_tokenizer(cfg)

    prompts = [args.prompt] if args.prompt else SAMPLE_PROMPTS

    for prompt in prompts:
        print(f"\n{'─'*60}")
        print(f"Pergunta: {prompt}")
        print("─" * 60)
        response = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"Resposta:\n{response}")

    print(f"\n{'─'*60}")


if __name__ == "__main__":
    main()
