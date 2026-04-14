"""
Passo 1: Geração de Dataset Sintético via API da OpenAI
Domínio: Assistente de Análise de Dados com Python

Este script usa o modelo GPT para criar pares de instrução/resposta
voltados ao ensino de análise de dados com pandas, numpy e matplotlib.
O resultado é dividido em 90% treino e 10% teste, salvo em .jsonl.
"""

import os
import json
import random
from openai import OpenAI

# ──────────────────────────────────────────────
# Configuração do cliente
# ──────────────────────────────────────────────
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

DOMAIN = "análise de dados com Python (pandas, numpy, matplotlib, scikit-learn)"

TOPICS = [
    "leitura e escrita de arquivos CSV com pandas",
    "limpeza de dados: tratamento de valores nulos",
    "filtragem e seleção de linhas/colunas em DataFrames",
    "agrupamento e agregação com groupby",
    "visualização de dados com matplotlib e seaborn",
    "merge e join entre DataFrames",
    "criação de novas colunas derivadas",
    "detecção e remoção de outliers",
    "normalização e padronização de features",
    "pipeline de pré-processamento com scikit-learn",
    "análise exploratória de dados (EDA)",
    "manipulação de datas e séries temporais",
    "operações vetorizadas com numpy",
    "salvamento e carregamento de modelos com joblib",
    "divisão treino/teste com train_test_split",
]

SYSTEM_PROMPT = (
    "Você é um especialista em ciência de dados que cria materiais educativos. "
    "Gere perguntas práticas e objetivas sobre {domain}, seguidas de respostas "
    "detalhadas com exemplos de código Python funcional. "
    "Responda SOMENTE com JSON válido no formato: "
    '{"prompt": "<pergunta>", "response": "<resposta completa com código>"}'
)


def generate_pair(topic: str) -> dict | None:
    """Gera um par prompt/response para um tópico dado."""
    user_msg = (
        f"Crie uma pergunta técnica sobre '{topic}' em {DOMAIN} "
        "e sua resposta completa com exemplo de código Python."
    )
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(domain=DOMAIN)},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.8,
                max_tokens=600,
            )
            raw = resp.choices[0].message.content.strip()
            pair = json.loads(raw)
            # Valida estrutura mínima
            if "prompt" in pair and "response" in pair:
                return pair
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [aviso] tentativa {attempt}/{max_retries} falhou: {e}")
        except Exception as e:
            print(f"  [erro API] tentativa {attempt}/{max_retries}: {e}")
            break   # erros de rede/auth não se resolvem com retry
    return None


def generate_dataset(target: int = 55) -> list[dict]:
    """Gera `target` pares válidos iterando sobre os tópicos."""
    dataset: list[dict] = []
    topics_cycle = (TOPICS * ((target // len(TOPICS)) + 2))[:target]
    random.shuffle(topics_cycle)

    print(f"Gerando {target} pares de instrução/resposta...")
    for i, topic in enumerate(topics_cycle, 1):
        pair = generate_pair(topic)
        if pair:
            dataset.append(pair)
            print(f"  [{i:>2}/{target}] OK — tópico: {topic[:50]}")
        else:
            print(f"  [{i:>2}/{target}] FALHOU — pulando tópico: {topic}")

    return dataset


def split_and_save(dataset: list[dict], train_ratio: float = 0.9) -> None:
    """Divide e salva os dados em arquivos .jsonl."""
    random.shuffle(dataset)
    split_idx = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    os.makedirs("data", exist_ok=True)

    for split_name, split_data in [("train", train_data), ("test", test_data)]:
        path = f"data/{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Salvo: {path} ({len(split_data)} exemplos)")


if __name__ == "__main__":
    random.seed(42)
    dataset = generate_dataset(target=55)

    if len(dataset) < 50:
        raise RuntimeError(
            f"Dataset insuficiente: apenas {len(dataset)} pares gerados "
            "(mínimo 50 exigido). Verifique sua chave de API e tente novamente."
        )

    split_and_save(dataset)
    print(f"\nDataset completo: {len(dataset)} pares totais.")
