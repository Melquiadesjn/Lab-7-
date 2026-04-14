"""
Avaliação Quantitativa do Modelo Fine-Tunado

Roda o modelo sobre o split de teste e calcula métricas de geração:
  - BLEU-4   : precisão de n-gramas (sobreposição com referência)
  - ROUGE-L  : cobertura da subsequência mais longa
  - Perplexidade : quão "surpreso" o modelo fica com os dados de teste

Uso: python evaluate.py
"""

import json
import math
import yaml
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ──────────────────────────────────────────────────────────────────────
# Utilitários de métricas (sem dependência de nltk/rouge-score)
# ──────────────────────────────────────────────────────────────────────

def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu4(reference: str, hypothesis: str) -> float:
    """BLEU-4 simplificado (sem brevity penalty para avaliação rápida)."""
    ref_tokens  = reference.lower().split()
    hyp_tokens  = hypothesis.lower().split()
    if not hyp_tokens:
        return 0.0

    score = 1.0
    for n in range(1, 5):
        ref_ng  = set(ngrams(ref_tokens, n))
        hyp_ng  = ngrams(hyp_tokens, n)
        if not hyp_ng:
            return 0.0
        matches = sum(1 for ng in hyp_ng if ng in ref_ng)
        precision = matches / len(hyp_ng)
        if precision == 0:
            return 0.0
        score *= precision

    return score ** (1 / 4)


def lcs_length(a: list, b: list) -> int:
    """Comprimento da maior subsequência comum."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i % 2][j] = dp[(i-1) % 2][j-1] + 1
            else:
                dp[i % 2][j] = max(dp[(i-1) % 2][j], dp[i % 2][j-1])
    return dp[m % 2][n]


def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L (F1 baseado em LCS)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall    = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ──────────────────────────────────────────────────────────────────────
# Carregamento do modelo
# ──────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["id"], quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, cfg["paths"]["adapter_dir"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["id"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def compute_perplexity(model, tokenizer, text: str) -> float:
    """Calcula a perplexidade do modelo sobre um texto de referência."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return math.exp(out.loss.item())


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    formatted = f"[INST] {prompt.strip()} [/INST]"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new = ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()

    with open(cfg["paths"]["test_file"], encoding="utf-8") as f:
        test_data = [json.loads(l) for l in f if l.strip()]

    print(f"Avaliando {len(test_data)} exemplos do split de teste...\n")
    print("Carregando modelo (aguarde)...")
    model, tokenizer = load_model(cfg)

    bleu_scores, rouge_scores, perp_scores = [], [], []

    for i, item in enumerate(test_data, 1):
        reference  = item["response"]
        hypothesis = generate(model, tokenizer, item["prompt"])
        perplexity = compute_perplexity(model, tokenizer, reference)

        b = bleu4(reference, hypothesis)
        r = rouge_l(reference, hypothesis)
        bleu_scores.append(b)
        rouge_scores.append(r)
        perp_scores.append(perplexity)

        print(f"Exemplo {i}/{len(test_data)}")
        print(f"  BLEU-4  : {b:.4f}")
        print(f"  ROUGE-L : {r:.4f}")
        print(f"  Perpl.  : {perplexity:.2f}\n")

    print("=" * 50)
    print("RESULTADOS MÉDIOS NO SPLIT DE TESTE")
    print("=" * 50)
    print(f"  BLEU-4    : {np.mean(bleu_scores):.4f}")
    print(f"  ROUGE-L   : {np.mean(rouge_scores):.4f}")
    print(f"  Perplexidade : {np.mean(perp_scores):.2f}")


if __name__ == "__main__":
    main()
