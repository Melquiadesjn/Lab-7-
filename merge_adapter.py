"""
Merge do Adapter LoRA no Modelo Base

Após o fine-tuning, os pesos do adapter ficam separados do modelo base.
Para deploy em produção (ou upload para a HuggingFace Hub), é necessário
mesclar os dois em um único modelo completo em fp16.

Esse script:
  1. Carrega o modelo base em fp16 (sem quantização, para merge correto)
  2. Aplica o adapter LoRA salvo
  3. Mescla os pesos (merge_and_unload)
  4. Salva o modelo completo em outputs/merged/

Uso: python merge_adapter.py
"""

import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_and_save(cfg: dict) -> None:
    model_id    = cfg["model"]["id"]
    adapter_dir = cfg["paths"]["adapter_dir"]
    merged_dir  = cfg["paths"]["merged_dir"]

    print(f"Carregando modelo base: {model_id}")
    print("(em fp16 — sem quantização para merge correto)\n")

    # Carrega em fp16 puro, sem BitsAndBytes
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Aplicando adapter de: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Realizando merge dos pesos (adapter → base)...")
    model = model.merge_and_unload()   # retorna modelo padrão sem camadas LoRA

    os.makedirs(merged_dir, exist_ok=True)

    print(f"Salvando modelo mesclado em: {merged_dir}")
    model.save_pretrained(merged_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)

    print("\nMerge concluído com sucesso!")
    print(f"Modelo pronto para deploy em: {merged_dir}")
    print("Para subir no HuggingFace Hub:")
    print(f"  huggingface-cli upload SEU_USUARIO/NOME_DO_REPO {merged_dir}")


if __name__ == "__main__":
    cfg = load_config()
    merge_and_save(cfg)
