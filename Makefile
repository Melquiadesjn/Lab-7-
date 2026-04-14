# Makefile — atalhos para o pipeline completo do Lab 07
# Uso: make <target>

.PHONY: install dataset validate train evaluate merge clean

## Instala as dependências do projeto
install:
	pip install -r requirements.txt

## Gera um novo dataset sintético via API da OpenAI
## Requer: export OPENAI_API_KEY="..."
dataset:
	python generate_dataset.py

## Valida os arquivos .jsonl antes de treinar
validate:
	python validate_dataset.py

## Executa o fine-tuning QLoRA completo
## Requer: GPU com VRAM >= 16 GB + acesso ao Llama-2 na HuggingFace
train: validate
	python train.py

## Avalia o modelo com BLEU-4, ROUGE-L e Perplexidade
evaluate:
	python evaluate.py

## Faz o merge do adapter LoRA no modelo base (para deploy)
merge:
	python merge_adapter.py

## Testa o modelo com perguntas de exemplo
inference:
	python inference.py

## Remove artefatos de treinamento (pesos, checkpoints)
clean:
	rm -rf outputs/
	@echo "Diretório outputs/ removido."

## Pipeline completo: instalar → gerar → validar → treinar → avaliar
all: install dataset validate train evaluate
	@echo "Pipeline completo finalizado."
