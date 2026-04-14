# Laboratório 07 — Especialização de LLMs com LoRA e QLoRA

> **Partes geradas/complementadas com IA, revisadas por Melquiades.**

Pipeline completo de fine-tuning do Llama 2 7B usando PEFT/LoRA e
quantização QLoRA, viabilizando o treinamento em GPUs com VRAM limitada.

---

## Domínio do Dataset

**Assistente de Análise de Dados com Python** — o modelo é especializado
para responder perguntas técnicas sobre pandas, numpy, matplotlib, seaborn
e scikit-learn, sempre com exemplos de código funcionais.

---

## Estrutura do Projeto

```
.
├── generate_dataset.py   # Passo 1 — gera dataset sintético via OpenAI API
├── train.py              # Passos 2-4 — pipeline QLoRA completo
├── requirements.txt      # Dependências do projeto
├── data/
│   ├── train.jsonl       # 45 exemplos de treino (90%)
│   └── test.jsonl        # 5 exemplos de teste (10%)
└── outputs/              # Criado durante o treinamento (não versionado)
    └── adapter/          # Pesos do adapter LoRA (salvo via save_pretrained)
```

---

## Como Reproduzir

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Gerar o dataset sintético (opcional — já incluído em `data/`)

```bash
export OPENAI_API_KEY="sua-chave-aqui"
python generate_dataset.py
```

### 3. Executar o fine-tuning

> Requer GPU com pelo menos 16 GB de VRAM e acesso ao Llama 2 na HuggingFace.

```bash
# Autenticar na HuggingFace (necessário para Llama 2)
huggingface-cli login

python train.py
```

O adapter treinado será salvo em `outputs/adapter/`.

---

## Decisões Técnicas

### Quantização (Passo 2)
O modelo base é carregado em **4-bit NF4** (NormalFloat 4-bit) via
`BitsAndBytesConfig`, reduzindo o uso de VRAM em ~4x. O `compute_dtype`
`float16` garante precisão adequada nas operações de forward/backward.

### LoRA (Passo 3)
Apenas as matrizes de atenção (`q_proj`, `k_proj`, `v_proj`, `o_proj`) são
adaptadas. Os hiperparâmetros seguem o roteiro:

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `r`       | 64    | Rank das matrizes de decomposição |
| `alpha`   | 16    | Fator de escala dos novos pesos |
| `dropout` | 0.1   | Regularização contra overfitting |

### Otimizador e Scheduler (Passo 4)
- **`paged_adamw_32bit`**: estados do AdamW em memória paginada, com
  transferência automática para CPU nos picos de uso.
- **Cosine scheduler**: decaimento suave da taxa de aprendizado.
- **`warmup_ratio=0.03`**: os primeiros 3% dos passos aquecen o LR
  gradualmente de 0 até o valor alvo.

---

## Dependências Principais

| Biblioteca       | Papel                                      |
|------------------|--------------------------------------------|
| `transformers`   | Carregamento do modelo base Llama 2        |
| `peft`           | Implementação do LoRA (`LoraConfig`)       |
| `trl`            | Treinamento supervisionado (`SFTTrainer`)  |
| `bitsandbytes`   | Quantização 4-bit (`BitsAndBytesConfig`)   |
| `datasets`       | Carregamento dos arquivos `.jsonl`         |
| `openai`         | Geração do dataset sintético               |
