"""
Validação do Dataset antes do treinamento

Verifica integridade estrutural dos arquivos .jsonl:
  - Campos obrigatórios presentes em todos os exemplos
  - Ausência de exemplos vazios ou com texto muito curto
  - Distribuição do tamanho dos textos
  - Ausência de duplicatas exatas no prompt

Uso: python validate_dataset.py
"""

import json
import sys
from pathlib import Path
from collections import Counter


REQUIRED_FIELDS = {"prompt", "response"}
MIN_PROMPT_LEN  = 10
MIN_RESPONSE_LEN = 20


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [ERRO] linha {i} inválida em '{path}': {e}")
    return records


def validate_split(path: str, split_name: str) -> bool:
    print(f"\n{'='*55}")
    print(f"  Validando split: {split_name} ({path})")
    print(f"{'='*55}")

    if not Path(path).exists():
        print(f"  [ERRO] Arquivo não encontrado: {path}")
        return False

    records = load_jsonl(path)
    if not records:
        print("  [ERRO] Nenhum registro encontrado.")
        return False

    errors   = []
    warnings = []
    prompt_lengths   = []
    response_lengths = []
    prompts_seen     = Counter()

    for i, rec in enumerate(records, 1):
        # Campos obrigatórios
        missing = REQUIRED_FIELDS - rec.keys()
        if missing:
            errors.append(f"  Exemplo {i}: campos ausentes {missing}")
            continue

        prompt   = str(rec["prompt"]).strip()
        response = str(rec["response"]).strip()

        if len(prompt) < MIN_PROMPT_LEN:
            errors.append(f"  Exemplo {i}: prompt muito curto ({len(prompt)} chars)")
        if len(response) < MIN_RESPONSE_LEN:
            errors.append(f"  Exemplo {i}: response muito curta ({len(response)} chars)")

        prompt_lengths.append(len(prompt))
        response_lengths.append(len(response))
        prompts_seen[prompt] += 1

    # Duplicatas
    duplicates = {p: c for p, c in prompts_seen.items() if c > 1}
    if duplicates:
        for p, c in duplicates.items():
            warnings.append(f"  Prompt duplicado ({c}x): '{p[:60]}...'")

    # Relatório
    total = len(records)
    print(f"  Total de exemplos : {total}")
    if prompt_lengths:
        print(f"  Prompt  — média: {sum(prompt_lengths)/len(prompt_lengths):.0f} chars"
              f"  | min: {min(prompt_lengths)}  | max: {max(prompt_lengths)}")
        print(f"  Response — média: {sum(response_lengths)/len(response_lengths):.0f} chars"
              f"  | min: {min(response_lengths)} | max: {max(response_lengths)}")

    for w in warnings:
        print(f"  [AVISO] {w}")
    for e in errors:
        print(f"  [ERRO]  {e}")

    if not errors:
        print(f"  [OK] {split_name} passou na validação.")
    else:
        print(f"  [FALHOU] {len(errors)} erro(s) encontrado(s).")

    return len(errors) == 0


def main() -> None:
    splits = [
        ("data/train.jsonl", "treino"),
        ("data/test.jsonl",  "teste"),
    ]

    all_ok = True
    for path, name in splits:
        ok = validate_split(path, name)
        all_ok = all_ok and ok

    print(f"\n{'='*55}")
    if all_ok:
        print("  Dataset válido — pronto para o treinamento.")
    else:
        print("  Dataset inválido — corrija os erros antes de treinar.")
        sys.exit(1)


if __name__ == "__main__":
    main()
