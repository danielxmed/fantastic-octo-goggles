# CS336 Spring 2025 — Assignment 1 Basics, versão Pindorama

> **Nota da versão (maio/2026)**: este markdown é uma transcrição fiel do notebook original, mantendo todo o código e a narrativa, com inserções pedagógicas e fact-checks alinhados ao estado da arte em maio de 2026. Os trechos em blockquote marcados como **Nota de 2026** ou **Intuição** são adições, não estavam no notebook original. O CS336 já tem uma edição **Spring 2026** ativa no campus de Stanford (lecionada entre março e junho de 2026); o repositório `assignment1-basics` recebeu commits até abril de 2026 e a estrutura central — BPE, RMSNorm, SwiGLU, RoPE, atenção causal — permanece a mesma do Spring 2025 que este notebook segue.

Este notebook é um roteiro de estudo executável para o primeiro assignment do CS336: construir as peças básicas de um pequeno language model causal. A mudança principal em relação ao enunciado original é trocar TinyStories/OpenWebText por `tylerxdurden/pindorama-corpus`, um corpus literário em português brasileiro/lusófono publicado no Hugging Face.

A intenção aqui não é entregar um atalho opaco. A intenção é criar um laboratório:

- primeiro você implementa as peças fundamentais de forma explícita;
- depois compara com as abstrações prontas do PyTorch e do ecossistema;
- por fim treina um modelo pequeno o bastante para rodar em Colab, mas organizado como um projeto que poderia crescer.

O notebook cobre a superfície real do Assignment 1: BPE, tokenizer, batches, Linear/Embedding, RMSNorm, SwiGLU, RoPE, atenção causal, Transformer LM, softmax/cross-entropy, AdamW, schedule, gradient clipping, checkpointing, loop de treino, geração e notas de produção.

---

## 0. Como usar este notebook

Use em três passadas.

1. **Passada conceitual**: leia os markdowns e execute apenas as células pequenas. O objetivo é entender shapes, invariantes e trade-offs.
2. **Passada de implementação**: apague ou esconda trechos de código e tente reimplementar. Compare com a versão abaixo e com os testes do assignment.
3. **Passada experimental**: rode o pipeline Pindorama, treine um modelo pequeno e mude uma coisa por vez: vocabulário, contexto, largura, profundidade, learning rate, batch efetivo.

O notebook foi escrito para Colab/H100, mas não assume GPU obrigatória. Em CPU ele roda os smoke tests e um treino mínimo; em GPU ele usa `bf16`, `scaled_dot_product_attention`, `torch.compile` quando disponível e `pin_memory` em DataLoader-like paths.

> **Nota de 2026** — *Hardware moderno*: H100/H200 (Hopper) e B200 (Blackwell) já são o padrão para LMs em escala média. O CS336 Spring 2026 oficialmente recomenda B200 via Modal ($6.25/h) para os runs de treino dos assignments. Em CPU, a passagem conceitual continua viável; em Colab gratuito (T4) você roda smoke tests confortavelmente. Para bf16, qualquer GPU Ampere+ (A100, RTX 30/40, L4) funciona.

```python
# Em Colab, rode esta célula uma vez. Em um ambiente local com uv/conda, você pode comentar.
import sys
import subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "tokenizers", "einops", "tqdm", "matplotlib"])
```

#### Explicação linha a linha

A célula serve apenas para garantir que as dependências externas existam quando o notebook é aberto no Google Colab, sem quebrar a execução em ambientes locais que já têm tudo instalado.

- `import sys`: importa o módulo padrão `sys` para inspecionar o estado do interpretador. A chave aqui é o atributo `sys.modules`, um dicionário com todos os módulos já carregados pelo Python no processo atual.
- `import subprocess`: módulo padrão para disparar comandos externos. Vamos usá-lo para chamar `pip` como um subprocesso, em vez de tentar importar `pip` diretamente (anti-pattern moderno; o time do `pip` desencoraja isso).
- `IN_COLAB = "google.colab" in sys.modules`: heurística canônica para detectar Colab. Quando o notebook abre dentro do Colab, o ambiente já injetou o módulo `google.colab` (responsável por upload de arquivos, autenticação Google etc.). Em Jupyter local, esse módulo não existe, então `IN_COLAB` será `False`.
- `if IN_COLAB:`: bloqueia a instalação para fora do Colab, evitando reinstalar pacotes em quem já tem o ambiente preparado.
- `subprocess.check_call([...])`: executa um comando e levanta `CalledProcessError` caso o exit code seja diferente de zero — útil para falhar barulhento se o `pip install` quebrar.
  - `sys.executable`: caminho absoluto do Python *do kernel atual*. Usar isso em vez de `"python"` evita o pitfall clássico de instalar no Python "errado" do sistema.
  - `"-m", "pip", "install"`: chama o módulo `pip` do Python correspondente; o `-m` é a forma idiomática.
  - `"-q"`: modo silencioso, reduz log no notebook.
  - `"datasets", "tokenizers", "einops", "tqdm", "matplotlib"`: respectivamente, biblioteca da Hugging Face para acessar datasets, biblioteca Rust de tokenização (BPE/WordPiece eficientes), DSL para reorganizar tensores, barra de progresso e plot 2D. Tudo o que o restante do notebook usa que não está em `numpy`/`torch`.

```python
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from tokenizers import ByteLevelBPETokenizer
except Exception:
    ByteLevelBPETokenizer = None

torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

print("device:", DEVICE)
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print("gpu:", props.name, "memory_gb:", round(props.total_memory / 1e9, 2), "bf16:", torch.cuda.is_bf16_supported())
print("torch:", torch.__version__)
```

#### Explicação linha a linha

Esta é a célula de "boilerplate" do notebook: imports, sementes de aleatoriedade e detecção de hardware.

- `from __future__ import annotations`: ativa o PEP 563 para postergar a avaliação de anotações de tipo. O efeito prático é deixar você escrever `dict[int, str]` ou `list[int] | None` mesmo em Python 3.9, sem precisar das classes do módulo `typing`. Em Python 3.10+ isso é redundante mas continua inofensivo, e funciona como sinal de portabilidade.
- `import json, math, os, random, time`: stdlib. `json` serializa dicionários para arquivos `.json` (vamos salvar configs); `math` traz `pi`, `sqrt`, `cos` etc.; `os` interage com o filesystem; `random` é o RNG da stdlib (separado do `numpy.random` e do `torch.random`); `time` mede latência de treino.
- `from collections import Counter, defaultdict`: `Counter` é um `dict` especializado para contar ocorrências (essencial no BPE — contar pares de bytes); `defaultdict` evita o pattern verboso de `if key not in d: d[key] = []`.
- `from dataclasses import dataclass, asdict`: `@dataclass` cria automaticamente `__init__`, `__repr__` e `__eq__` a partir de anotações de tipo. `asdict` converte uma dataclass para um dicionário serializável — útil para checkpoint.
- `from pathlib import Path`: API orientada a objeto para caminhos de arquivo. `Path("a") / "b"` é mais limpo do que `os.path.join("a", "b")`.
- `from typing import Iterable, Iterator, Optional`: tipos abstratos. `Iterable` é "qualquer coisa que pode ser iterada"; `Iterator` é o protocolo `__next__`; `Optional[X]` é açúcar para `X | None`.
- `import numpy as np`: NumPy para tokenização (arrays compactos `uint16` ocupam metade da memória de `int32`) e para o RNG.
- `import torch / torch.nn as nn / torch.nn.functional as F`: o tripé canônico. `torch` traz tensores, `nn` traz módulos com estado (parâmetros), `F` traz funções puras (sem estado).
- `from torch import Tensor`: importa o tipo `Tensor` para usar em type hints — `def f(x: Tensor) -> Tensor` é mais legível do que `torch.Tensor`.
- `from torch.utils.data import IterableDataset, DataLoader`: estruturas para *streaming* de dados. Não vamos usar pesado neste notebook (preferimos `get_batch` direto sobre `np.ndarray`), mas seriam o caminho de produção.
- `try / except` em torno de `from datasets import load_dataset`: tolera ausência da biblioteca `datasets`. Se ela não estiver instalada, `load_dataset = None` e o resto do código verifica isso explicitamente. Mesma lógica para `tokenizers`/`ByteLevelBPETokenizer`. Esse padrão de "import opcional" é o que permite o notebook rodar offline com fallback.
- `torch.manual_seed(1337)`: fixa a semente do RNG global do PyTorch (CPU). Determina inicializações de pesos, dropout, `randint` etc. O `1337` é folclore — qualquer inteiro funciona.
- `np.random.seed(1337)`: fixa a semente do NumPy. Útil porque tokenização e splits manuais podem usar `np.random`.
- `random.seed(1337)`: fixa a semente do `random` da stdlib. Tokenizadores Python e shuffles em `list` chamam isso.
- `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`: detecta GPU. Se `torch` foi instalado com suporte CUDA *e* a máquina tem GPU NVIDIA acessível, usa "cuda"; caso contrário, "cpu". Em Apple Silicon você poderia trocar por `"mps"`.
- `DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float32`: bf16 (Brain Float 16) tem o mesmo *expoente* de fp32 mas mantissa de 7 bits — é o dtype canônico de treino moderno. Ampere (A100, RTX 30+) e mais novas suportam; Volta (V100) não. Em CPU, ficamos em fp32.
- `print("device:", DEVICE)`: log básico para confirmar onde estamos.
- `if DEVICE == "cuda": props = torch.cuda.get_device_properties(0)`: busca metadados da GPU 0 (a primeira). `props.name` é coisa como `"NVIDIA A100-SXM4-80GB"`, `props.total_memory` em bytes.
- `round(props.total_memory / 1e9, 2)`: converte para gigabytes legíveis (com 2 decimais).
- `print("torch:", torch.__version__)`: registra a versão de PyTorch — essencial para reprodutibilidade quando alguém ler o notebook seis meses depois.

> **Intuição — três sementes**: `torch.manual_seed`, `np.random.seed` e `random.seed` cobrem três fontes independentes de não-determinismo. Quase nada em PyTorch toca `random` (Python stdlib), mas tokenizers, splits manuais e shufflers Python costumam tocar. Em GPU, ainda há não-determinismo residual em kernels CUDA (atomic adds, etc.) — para reprodução bit-a-bit, adicione `torch.use_deterministic_algorithms(True)` e `CUBLAS_WORKSPACE_CONFIG=:4096:8`, sabendo que isso custa performance.

---

## 1. O que o Assignment 1 realmente pede

O assignment não é apenas "treinar um GPT pequeno". Ele força você a atravessar a pilha inteira:

- **Dados e tokenização**: treinar BPE byte-level, preservar tokens especiais, implementar encode/decode, criar batches `x -> y`.
- **Camadas**: Linear sem bias, Embedding, RMSNorm, SiLU/SwiGLU, RoPE, atenção escalada, MHA causal.
- **Modelo**: bloco Transformer pre-norm, LM causal completo, logits por posição.
- **Otimização**: softmax estável, cross-entropy, AdamW, cosine schedule com warmup, clipping.
- **Engenharia**: checkpoint save/load, loop de treino, avaliação, geração, desempenho e reprodutibilidade.

O foco do CS336 é "entender construindo". Por isso muitas implementações abaixo são explícitas mesmo quando PyTorch tem uma versão pronta. A seção "produção" de cada módulo mostra quando você deveria trocar código manual por uma abstração otimizada.

> **Nota de 2026 — porquê isso ainda importa**: a justificativa do CS336 é que abstrações vazam. Um exemplo concreto da edição 2026 do curso: a fração de FLOPs gasta em MLP cresce de ~44% num modelo pequeno para ~80% num modelo de 175B. Quem só usou `nn.Linear` por anos sem entender essa quebra fica perdido quando precisa decidir entre tensor parallelism em FFN vs sharding em attention. O assignment 1 é a base; os assignments 2-5 (systems, scaling, data, alignment) crescem dali.

---

## 2. Pindorama Corpus: dados literários em português

O Pindorama Corpus é particularmente bom para este estudo por três motivos.

Primeiro, ele torna a tokenização menos trivial: acentos, cedilha, hífens, travessões, grafia histórica e nomes próprios aparecem com frequência. Um tokenizer word-level ficaria frágil; um byte-level BPE continua total, isto é, sempre consegue representar qualquer string UTF-8.

Segundo, a distribuição é diferente de TinyStories. A validação didática fica mais interessante porque literatura adulta tem períodos longos, pontuação rica, vocabulário variado e estilos heterogêneos.

Terceiro, o dataset já traz metadados úteis de curadoria: `is_recommended_for_pretraining`, deduplicação e flags de qualidade. Mesmo em assignment, vale treinar a disciplina de não tratar "texto" como uma massa sem proveniência.

> **Intuição — por que português é teste duro**: em inglês, um BPE de 32k cobre quase tudo com tokens de 4-6 bytes em média. Em português, "construção", "transformação", "psicologicamente" e nomes próprios brasileiros (geográficos, indígenas, africanos) puxam a distribuição para cauda longa. Na prática você verá `tokens/byte` mais alto em PT que em EN para o mesmo vocab_size — uma proxy útil para "o tokenizer está comprimindo bem este idioma?". Petrov et al. (2023) e estudos posteriores documentam essa "iniquidade multilíngue" como artefato de tokenização, não do modelo.

```python
DATASET_NAME = "tylerxdurden/pindorama-corpus"
SPECIAL_TOKEN = "<|endoftext|>"
WORKDIR = Path("pindorama_assignment1_work")
WORKDIR.mkdir(exist_ok=True)

FALLBACK_TEXTS = [
    "No meio do caminho tinha uma pedra. Tinha uma pedra no meio do caminho.",
    "A literatura brasileira guarda vozes, ritmos, cidades, sertoes e memorias.",
    "O estudante observa os tensores, mede a perda e ajusta o passo do otimizador.",
    "Era uma vez uma biblioteca publica cheia de poemas, romances e cronicas.",
]


def load_pindorama(max_train_docs: int = 256, max_valid_docs: int = 64):
    '''Carrega um subconjunto pequeno e reprodutivel para estudo.

    Em Colab, o Hugging Face baixa Parquet sob demanda. Se a rede falhar,
    usamos um fallback pequeno para manter o notebook executavel.
    '''
    if load_dataset is None:
        print("datasets nao instalado; usando fallback local.")
        return FALLBACK_TEXTS, FALLBACK_TEXTS[:2]
    try:
        ds = load_dataset(DATASET_NAME)
        train = ds["train"]
        valid = ds["validation"] if "validation" in ds else ds["train"]
        if "is_recommended_for_pretraining" in train.column_names:
            train = train.filter(lambda row: bool(row["is_recommended_for_pretraining"]))
        train_texts = [row["text"] for row in train.select(range(min(max_train_docs, len(train)))) if row.get("text")]
        valid_texts = [row["text"] for row in valid.select(range(min(max_valid_docs, len(valid)))) if row.get("text")]
        print("train docs:", len(train_texts), "valid docs:", len(valid_texts))
        return train_texts, valid_texts
    except Exception as exc:
        print("Falha ao carregar Hugging Face; usando fallback local.")
        print(type(exc).__name__ + ":", exc)
        return FALLBACK_TEXTS, FALLBACK_TEXTS[:2]


train_texts, valid_texts = load_pindorama()
print(train_texts[0][:500])
```

#### Explicação linha a linha

Esta célula define como obter o corpus Pindorama de forma resiliente — funciona online, com fallback offline.

- `DATASET_NAME = "tylerxdurden/pindorama-corpus"`: identificador na Hugging Face Hub. O formato é `usuario/dataset` (ou `org/dataset`).
- `SPECIAL_TOKEN = "<|endoftext|>"`: o token especial herdado do GPT-2 que marca fim de documento. Manter o nome canônico ajuda a interoperar com tokenizadores prontos.
- `WORKDIR = Path("pindorama_assignment1_work")`: pasta única para todos os artefatos do notebook. Concentrar saídas em uma pasta facilita limpar.
- `WORKDIR.mkdir(exist_ok=True)`: cria o diretório se não existe. `exist_ok=True` evita erro se já houver.
- `FALLBACK_TEXTS = [...]`: quatro frases curtas em português usadas quando o download da Hugging Face falha. Não é representativo do corpus real, mas permite que cada célula seguinte continue executável (smoke test).
- `def load_pindorama(max_train_docs: int = 256, max_valid_docs: int = 64):`: a assinatura usa argumentos com default pequenos para que o notebook seja rápido. Você pode aumentá-los depois.
- `'''Carrega um subconjunto pequeno e reprodutivel para estudo. ...'''`: docstring tripla. Boas práticas: documente *o que* a função faz, não *como*.
- `if load_dataset is None:`: se a biblioteca `datasets` não foi importada (`try/except` na célula anterior), a função aceita derrota cedo e devolve o fallback.
- `try: ds = load_dataset(DATASET_NAME)`: baixa o dataset (Parquet sob demanda). Em Colab, é cacheado na primeira execução.
- `train = ds["train"]`: pega a partição de treino. Datasets na Hub seguem a convenção de splits "train"/"validation"/"test".
- `valid = ds["validation"] if "validation" in ds else ds["train"]`: alguns datasets só têm `train`; nesse caso a validação reusa `train` (longe do ideal, mas mantém o notebook rodando — você deveria fazer um split manual em produção).
- `if "is_recommended_for_pretraining" in train.column_names:`: o Pindorama traz uma flag de curadoria. Se ela existe, filtramos para o subconjunto recomendado.
- `train = train.filter(lambda row: bool(row["is_recommended_for_pretraining"]))`: aplica o filtro. `Dataset.filter` é lazy e cacheado pela `datasets`.
- `train_texts = [row["text"] for row in train.select(range(min(max_train_docs, len(train)))) if row.get("text")]`: pega os primeiros `max_train_docs` documentos, descartando linhas com texto vazio. `train.select(range(...))` é a forma idiomática de fatiar um `Dataset` lazy.
- Linha análoga para `valid_texts`.
- `print("train docs:", len(train_texts), "valid docs:", len(valid_texts))`: log de sanidade.
- `return train_texts, valid_texts`: retorna duas listas de strings.
- `except Exception as exc:` ... `print(...)`: captura qualquer falha (rede caída, dataset privado, schema mudou) e cai para o fallback. Exceção genérica é justificada em código de notebook didático.
- `return FALLBACK_TEXTS, FALLBACK_TEXTS[:2]`: usa as 4 frases para treino e as 2 primeiras para validação.
- `train_texts, valid_texts = load_pindorama()`: chama a função com defaults.
- `print(train_texts[0][:500])`: imprime os primeiros 500 caracteres do primeiro documento — uma checagem visual rápida de que o texto vem em UTF-8 com acentos preservados.

```python
def write_plaintext_corpus(texts: list[str], path: Path, special_token: str = SPECIAL_TOKEN) -> None:
    '''Escreve um arquivo texto com separador de documento explicito.'''
    with path.open("w", encoding="utf-8") as f:
        for text in texts:
            cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
            if cleaned:
                f.write(cleaned)
                f.write("\n")
                f.write(special_token)
                f.write("\n")


train_txt = WORKDIR / "pindorama_train_sample.txt"
valid_txt = WORKDIR / "pindorama_valid_sample.txt"
write_plaintext_corpus(train_texts, train_txt)
write_plaintext_corpus(valid_texts, valid_txt)
print(train_txt, train_txt.stat().st_size, "bytes")
```

#### Explicação linha a linha

A célula materializa os textos como arquivos `.txt` em UTF-8 — formato que a biblioteca `tokenizers` aceita como entrada de treino.

- `def write_plaintext_corpus(texts: list[str], path: Path, special_token: str = SPECIAL_TOKEN) -> None:`: assinatura tipada. Recebe uma lista de strings, um caminho de saída e o separador de documentos.
- `with path.open("w", encoding="utf-8") as f:`: abre o arquivo em modo "write" texto, com encoding explícito. *Sempre* declare encoding em arquivos de texto — o default de Python varia por sistema operacional.
- `for text in texts:`: itera por documento.
- `cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()`: normaliza quebras de linha. Windows usa `\r\n`, Mac antigo usa `\r`, Unix usa `\n`. Convertemos tudo para `\n`. `.strip()` remove espaços e quebras nas pontas.
- `if cleaned:`: pula documentos vazios após a limpeza.
- `f.write(cleaned)`: escreve o conteúdo do documento.
- `f.write("\n")`: separa documento do delimitador com uma quebra.
- `f.write(special_token)`: escreve `<|endoftext|>`. Esse marcador é como o tokenizer de produção saberá onde termina cada documento (importante para BPE não fundir tokens entre documentos).
- `f.write("\n")`: nova linha após o token, pronto para o próximo documento.
- `train_txt = WORKDIR / "pindorama_train_sample.txt"`: monta o caminho usando o operador `/` do `Path`.
- `valid_txt = WORKDIR / "pindorama_valid_sample.txt"`: idem para validação.
- `write_plaintext_corpus(train_texts, train_txt)`: dispara a escrita.
- `write_plaintext_corpus(valid_texts, valid_txt)`: idem para o split de validação.
- `print(train_txt, train_txt.stat().st_size, "bytes")`: imprime caminho e tamanho. `Path.stat()` retorna metadata; `.st_size` é o tamanho em bytes — útil para depois calcular a razão "tokens/byte" do tokenizer.

---

## 3. Byte-level BPE: teoria operacional

Um tokenizer precisa equilibrar três pressões:

- **Cobertura**: qualquer texto precisa virar IDs. Byte-level BPE resolve isso começando pelos 256 bytes possíveis.
- **Compressão**: sequências longas demais tornam atenção quadrática cara. BPE aprende merges frequentes para reduzir comprimento.
- **Generalização morfológica**: em português, formas como `casa`, `casas`, `casinha`, `casarao`, `educacao`, `educacional` compartilham pedaços úteis. BPE não entende morfologia, mas captura regularidades estatísticas.

O algoritmo de treino é simples: inicialize o vocabulário com bytes; conte pares adjacentes; funda o par mais frequente; repita. O problema de engenharia é fazer isso sem recalcular o mundo inteiro a cada merge quando o corpus cresce.

A implementação didática abaixo é correta para estudo e corpus pequeno. Para treinar o tokenizer real do experimento, usaremos o backend Rust da biblioteca `tokenizers`.

> **Nota de 2026 — onde BPE está hoje**: o estado da arte se movimentou de três formas relevantes desde 2024.
>
> 1. **SuperBPE** (Liu et al., COLM 2025): BPE em duas passagens — primeiro aprende subwords padrão, depois "superwords" que cruzam fronteiras de espaço em branco. Resultado: ~33% menos tokens, ganho médio de 4% em 30 benchmarks (8.2% em MMLU).
> 2. **BoundlessBPE** (COLM 2025): relaxa a restrição de pre-tokenization por whitespace, permitindo merges que cruzam essa fronteira diretamente. ~15% de melhoria em bytes/token.
> 3. **BLT — Byte Latent Transformer** (Pagnoni et al., Meta 2024, validado em escala 2025): primeira arquitetura *tokenizer-free* a igualar tokenization-based em escala (8B params, 4T bytes). Agrupa bytes em "patches" de tamanho dinâmico via entropia. Promessa: até 50% menos FLOPs de inferência e robustez nativa a typos, ortografia e iniquidade multilíngue.
>
> Para o assignment, BPE byte-level continua canônico — é o que GPT-2/3/4, Llama 2, e a maioria dos OSS usa. Mas a discussão "BPE forever?" deixou de ser retórica. Para PT-BR especificamente, SuperBPE é interessante porque "de a", "no meio do", "por isso" são unidades funcionais que BPE clássico nunca funde.

```python
def bytes_to_unicode() -> dict[int, str]:
    '''Mapeamento reversivel GPT-2 entre bytes e caracteres unicode visiveis.'''
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def get_pair_counts(token_ids: list[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(token_ids, token_ids[1:]))


def merge_pair(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(token_ids):
        if i + 1 < len(token_ids) and (token_ids[i], token_ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(token_ids[i])
            i += 1
    return out


def train_toy_byte_bpe(text: str, vocab_size: int = 320, special_tokens: list[str] | None = None):
    '''Treinador BPE pequeno para entender o mecanismo.

    Retorna vocab `id -> bytes` e lista de merges em bytes.
    '''
    special_tokens = special_tokens or []
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    token_ids = list(text.encode("utf-8"))
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        counts = get_pair_counts(token_ids)
        if not counts:
            break
        best_pair, freq = counts.most_common(1)[0]
        if freq < 2:
            break
        new_id = len(vocab)
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        token_ids = merge_pair(token_ids, best_pair, new_id)
    return vocab, merges


toy_vocab, toy_merges = train_toy_byte_bpe("\n".join(FALLBACK_TEXTS), vocab_size=280, special_tokens=[SPECIAL_TOKEN])
print("vocab:", len(toy_vocab), "merges:", len(toy_merges))
print("primeiros merges:", toy_merges[:10])
```

#### Explicação linha a linha

Esta célula reproduz, em Python puro, o algoritmo BPE byte-level do GPT-2. Vale ler com calma — internalize o algoritmo aqui antes de confiar na implementação Rust.

**Função `bytes_to_unicode()`**:
- `bs = list(range(ord("!"), ord("~") + 1)) + ...`: monta a lista de bytes que já são *imprimíveis* em UTF-8/Latin-1: `!` a `~` (33-126), e duas faixas extras de Latin-1 supplement (`¡`-`¬` e `®`-`ÿ`). São 188 bytes "bonitos" que podem ir direto para a representação visível.
- `cs = bs[:]`: faz uma cópia que será o "lado direito" do mapping (caracteres unicode visíveis).
- `n = 0`: contador para os bytes "feios" que precisam ser remapeados.
- `for b in range(256):`: percorre todos os 256 bytes possíveis.
- `if b not in bs:`: se o byte não está entre os imprimíveis…
- `bs.append(b); cs.append(256 + n); n += 1`: adiciona o byte original em `bs` e mapeia para `chr(256 + n)` — codepoints unicode acima de 256, garantidamente visíveis e fora do conflito ASCII.
- `return dict(zip(bs, map(chr, cs)))`: cria o dicionário `byte → caractere unicode`. Esse truque é o que torna `tokens.json` legível em qualquer editor (em vez de mostrar `\x00`, `\x01` etc.).

**Função `get_pair_counts(token_ids)`**:
- `Counter(zip(token_ids, token_ids[1:]))`: gera todos os pares adjacentes `(token_ids[i], token_ids[i+1])` usando o truque clássico de comprimir com `zip(seq, seq[1:])`. `Counter` conta as ocorrências de cada par. É o coração estatístico do BPE.

**Função `merge_pair(token_ids, pair, new_id)`**:
- `out: list[int] = []`: buffer de saída.
- `i = 0; while i < len(token_ids):`: índice manual em vez de `for` porque às vezes saltamos 2 posições.
- `if i + 1 < len(token_ids) and (token_ids[i], token_ids[i + 1]) == pair:`: verifica se há pelo menos um próximo elemento e se o par bate com `pair`.
- `out.append(new_id); i += 2`: substitui o par pelo novo ID e pula 2 posições.
- `else: out.append(token_ids[i]); i += 1`: mantém o token original e avança 1.
- `return out`: nova lista com o par fundido.

**Função `train_toy_byte_bpe(text, vocab_size, special_tokens)`**:
- `special_tokens = special_tokens or []`: aceita `None` substituindo por lista vazia.
- `vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}`: inicializa com os 256 bytes (cada ID começa apontando para o byte original como `bytes` de tamanho 1).
- `for tok in special_tokens: vocab[len(vocab)] = tok.encode("utf-8")`: adiciona special tokens *depois* dos 256 bytes para que IDs 0-255 sejam sempre bytes literais.
- `token_ids = list(text.encode("utf-8"))`: converte o texto em lista de inteiros 0-255 (UTF-8 bytes). É a representação inicial do corpus.
- `merges: list[tuple[bytes, bytes]] = []`: lista ordenada de merges. A *ordem* é crucial: o tokenizer aplica os merges nesta ordem na hora do encode.
- `while len(vocab) < vocab_size:`: loop principal — para quando o vocab atinge o tamanho desejado.
- `counts = get_pair_counts(token_ids)`: conta pares atuais.
- `if not counts: break`: se não há mais pares (corpus de 1 token), sai.
- `best_pair, freq = counts.most_common(1)[0]`: pega o par mais frequente. `most_common(n)` retorna lista de tuplas `(item, count)`; pegamos o primeiro elemento.
- `if freq < 2: break`: se o par mais frequente aparece só uma vez, fundir não comprime — pare.
- `new_id = len(vocab)`: o próximo ID livre.
- `merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))`: registra o merge em forma de tupla de `bytes`.
- `vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]`: o novo token é a concatenação dos dois bytes/sequências.
- `token_ids = merge_pair(token_ids, best_pair, new_id)`: aplica o merge e refaz a lista.
- `return vocab, merges`: devolve vocab e merges.

**Bloco final (chamada e prints)**:
- `toy_vocab, toy_merges = train_toy_byte_bpe(...)`: treina sobre as frases de fallback concatenadas. Aqui o corpus é minúsculo, mas o suficiente para gerar uma dezena de merges.
- `print("vocab:", len(toy_vocab), ...)`: confirma o tamanho final do vocabulário.
- `print("primeiros merges:", toy_merges[:10])`: imprime os 10 primeiros merges. Em PT-BR esperamos pares como `(b'a', b' ')`, `(b'e', b' ')`, `(b'd', b'e')` (todas frequentes em "de").

### Por que a versão acima não é produção

Ela varre a sequência inteira a cada merge. Isso é aceitável para uma célula didática, mas fica ruim quando o corpus chega a centenas de megabytes. O assignment cobra uma versão eficiente o bastante para os testes; em produção você normalmente usa `tokenizers`, `sentencepiece` ou `tiktoken`.

No CS336, a parte valiosa de implementar à mão é internalizar as invariantes:

- special tokens não devem ser quebrados em pedaços;
- bytes tornam o tokenizer fechado sob UTF-8;
- merges são ordenados, e essa ordem define a codificação;
- decode precisa ser reversível, mesmo quando aparecem bytes que não formam UTF-8 perfeito isoladamente.

> **Intuição — complexidade do BPE ingênuo**: o loop acima é O(n × k) onde n é o número de tokens no corpus e k o número de merges. Com 100M de tokens e 32k merges, isso é 3 × 10¹² operações — minutos no melhor caso. A versão Rust do `tokenizers` mantém uma estrutura de dados que rastreia apenas os pares afetados pelo último merge: O((n + k) × log n) amortizado. É a mesma ideia algorítmica de Huffman incremental.

```python
TOKENIZER_DIR = WORKDIR / "tokenizer"
TOKENIZER_DIR.mkdir(exist_ok=True)


def train_production_tokenizer(files: list[str], vocab_size: int = 8192):
    '''Treina ByteLevelBPETokenizer usando backend Rust.

    8192 e pequeno para Colab/assignment. Para experimentos melhores com 220M tokens,
    teste 16k, 32k e compare tokens/byte e valid loss.
    '''
    if ByteLevelBPETokenizer is None:
        raise RuntimeError("Instale tokenizers para treinar a versao de producao.")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[SPECIAL_TOKEN],
    )
    tokenizer.save_model(str(TOKENIZER_DIR))
    return tokenizer


if ByteLevelBPETokenizer is not None:
    production_tokenizer = train_production_tokenizer([str(train_txt)], vocab_size=4096)
    sample = "No meio do caminho havia uma pedra, uma memoria e um tensor."
    enc = production_tokenizer.encode(sample)
    print(enc.ids[:30])
    print(production_tokenizer.decode(enc.ids))
else:
    production_tokenizer = None
    print("tokenizers indisponivel.")
```

#### Explicação linha a linha

Esta célula treina o tokenizer "de produção" usando o backend Rust de `tokenizers` — o mesmo que GPT-2 e Llama usam por baixo de cobertura HuggingFace.

- `TOKENIZER_DIR = WORKDIR / "tokenizer"`: subpasta para os artefatos do tokenizer (`vocab.json` e `merges.txt`).
- `TOKENIZER_DIR.mkdir(exist_ok=True)`: cria se não existir.
- `def train_production_tokenizer(files: list[str], vocab_size: int = 8192):`: assinatura. `files` é uma lista de caminhos absolutos em string (a API do `tokenizers` quer `str`, não `Path`).
- `'''Treina ByteLevelBPETokenizer ...'''`: docstring com nota sobre o tamanho do vocab.
- `if ByteLevelBPETokenizer is None: raise RuntimeError(...)`: se a biblioteca não foi importada com sucesso, falha alto. (A função só deve ser chamada quando ela existe.)
- `tokenizer = ByteLevelBPETokenizer()`: instancia. Internamente, ele já conhece o pre-tokenizer "ByteLevel" (que mapeia bytes para o unicode visível, igual ao que `bytes_to_unicode` faz manualmente).
- `tokenizer.train(...)`: treina BPE. Argumentos:
  - `files=files`: lista de arquivos `.txt`.
  - `vocab_size=vocab_size`: parada do BPE.
  - `min_frequency=2`: descarta merges que aparecem só uma vez no corpus, evitando ruído.
  - `special_tokens=[SPECIAL_TOKEN]`: garante que `<|endoftext|>` ganha um ID e nunca é quebrado.
- `tokenizer.save_model(str(TOKENIZER_DIR))`: persiste `vocab.json` e `merges.txt` na pasta. `save_model` quer `str`, daí o cast.
- `return tokenizer`: devolve a instância já treinada e em memória.

**Bloco condicional final**:
- `if ByteLevelBPETokenizer is not None:`: tenta a versão de produção apenas se a biblioteca existe.
- `production_tokenizer = train_production_tokenizer([str(train_txt)], vocab_size=4096)`: treina com vocab pequeno (4096 — adequado para o corpus minúsculo).
- `sample = "No meio do caminho havia uma pedra, uma memoria e um tensor."`: frase de teste.
- `enc = production_tokenizer.encode(sample)`: codifica para um objeto `Encoding` que contém `ids`, `tokens`, `offsets` etc.
- `print(enc.ids[:30])`: primeiros 30 IDs.
- `print(production_tokenizer.decode(enc.ids))`: confirma roundtrip — decode deve devolver a string original.
- `else: production_tokenizer = None; print("tokenizers indisponivel.")`: marca ausência para o resto do notebook escolher o fallback.

```python
class SimpleByteTokenizer:
    '''Fallback reversivel: 256 bytes + special token.

    Ele nao comprime como BPE, mas permite executar o resto do notebook sem rede
    ou dependencias extras. O modelo fica mais lento porque a sequencia cresce.
    '''
    def __init__(self, special_token: str = SPECIAL_TOKEN):
        self.special_token = special_token
        self.special_id = 256
        self.vocab_size = 257

    def encode(self, text: str):
        class Enc:
            def __init__(self, ids):
                self.ids = ids
        ids: list[int] = []
        parts = text.split(self.special_token)
        for i, part in enumerate(parts):
            ids.extend(part.encode("utf-8"))
            if i != len(parts) - 1:
                ids.append(self.special_id)
        return Enc(ids)

    def decode(self, ids: list[int]) -> str:
        chunks: list[str] = []
        buf = bytearray()
        for idx in ids:
            if idx == self.special_id:
                if buf:
                    chunks.append(bytes(buf).decode("utf-8", errors="replace"))
                    buf.clear()
                chunks.append(self.special_token)
            else:
                buf.append(int(idx))
        if buf:
            chunks.append(bytes(buf).decode("utf-8", errors="replace"))
        return "".join(chunks)

    def token_to_id(self, token: str) -> int | None:
        return self.special_id if token == self.special_token else None


tokenizer = production_tokenizer if production_tokenizer is not None else SimpleByteTokenizer()
VOCAB_SIZE = tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else tokenizer.vocab_size
print("VOCAB_SIZE:", VOCAB_SIZE)
```

#### Explicação linha a linha

`SimpleByteTokenizer` é o tokenizer de fallback: 256 IDs para os bytes UTF-8 + 1 ID extra para o token especial. Não comprime, mas é *fechado* (sempre roundtripa qualquer string).

- `class SimpleByteTokenizer:`: classe simples, não herda de nada.
- `'''Fallback reversivel: 256 bytes + special token. ...'''`: docstring explica trade-off.
- `def __init__(self, special_token: str = SPECIAL_TOKEN):`: construtor. Aceita um token especial customizável.
- `self.special_token = special_token`: guarda a string.
- `self.special_id = 256`: o byte 256 não existe (bytes vão de 0 a 255), então 256 é livre para o special.
- `self.vocab_size = 257`: 256 bytes + 1 special.
- `def encode(self, text: str):`: emula a API de `tokenizers`, retornando um objeto com atributo `.ids`.
- `class Enc: def __init__(self, ids): self.ids = ids`: mini-classe local — duck typing para parecer um `Encoding`.
- `ids: list[int] = []`: buffer.
- `parts = text.split(self.special_token)`: divide o texto por ocorrências do special. Se o special está no texto literalmente, ele não vai virar bytes.
- `for i, part in enumerate(parts):`: itera com índice.
- `ids.extend(part.encode("utf-8"))`: adiciona os bytes UTF-8 do pedaço.
- `if i != len(parts) - 1: ids.append(self.special_id)`: entre pedaços (mas não depois do último), insere o ID do special.
- `return Enc(ids)`: empacota.
- `def decode(self, ids: list[int]) -> str:`: o inverso.
- `chunks: list[str] = []`: pedaços decodificados.
- `buf = bytearray()`: buffer mutável de bytes pendentes.
- `for idx in ids:`: percorre IDs.
- `if idx == self.special_id:`: encontrou special.
- `if buf: chunks.append(bytes(buf).decode("utf-8", errors="replace")); buf.clear()`: drena o buffer atual decodificando UTF-8 — `errors="replace"` substitui sequências inválidas por `�` em vez de falhar.
- `chunks.append(self.special_token)`: insere o token especial como string literal.
- `else: buf.append(int(idx))`: byte normal — acumula no buffer.
- `if buf: chunks.append(bytes(buf).decode("utf-8", errors="replace"))`: drena o buffer ao final.
- `return "".join(chunks)`: monta a string final.
- `def token_to_id(self, token: str) -> int | None:`: API mínima para procurar IDs por nome; só conhece o special.
- `return self.special_id if token == self.special_token else None`: simples.

**Seleção do tokenizer ativo**:
- `tokenizer = production_tokenizer if production_tokenizer is not None else SimpleByteTokenizer()`: prefere o BPE de produção quando disponível; cai para o byte-level puro caso contrário.
- `VOCAB_SIZE = tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else tokenizer.vocab_size`: o BPE da `tokenizers` expõe `get_vocab_size()` (método); o `SimpleByteTokenizer` expõe `vocab_size` (atributo). `hasattr` resolve a discordância.
- `print("VOCAB_SIZE:", VOCAB_SIZE)`: confirma o número final, que é usado por todas as camadas downstream (Embedding, lm_head).

---

## 4. Dataset tokenizado e batches `x -> y`

Language modeling causal é previsão do próximo token. Se a sequência tokenizada é

`[t0, t1, t2, t3, ...]`

um exemplo de contexto 4 vira:

- `x = [t0, t1, t2, t3]`
- `y = [t1, t2, t3, t4]`

O detalhe importante é amostrar índices iniciais válidos. Se o dataset tem `N` tokens e o contexto tem `T`, o maior início possível é `N - T - 1`, porque precisamos também do token alvo imediatamente depois da janela.

> **Intuição — paralelismo do shift**: em cada janela, o modelo prediz `T` tokens *em paralelo* (uma predição por posição, com mascaramento causal). Isto é, treinar com batch_size=B e contexto T processa `B × T` predições por step. É por isso que aumentar contexto é "grátis em paralelismo" mas "caro em quadrático" — atenção custa O(T²) por bloco, mas o gradiente flui por T predições.

```python
def encode_texts(texts: list[str], tokenizer) -> np.ndarray:
    ids: list[int] = []
    eot = tokenizer.token_to_id(SPECIAL_TOKEN) if hasattr(tokenizer, "token_to_id") else None
    for text in texts:
        ids.extend(tokenizer.encode(text).ids)
        if eot is not None:
            ids.append(eot)
    dtype = np.uint16 if VOCAB_SIZE <= np.iinfo(np.uint16).max else np.int32
    return np.asarray(ids, dtype=dtype)


train_ids = encode_texts(train_texts, tokenizer)
valid_ids = encode_texts(valid_texts, tokenizer)
print("train tokens:", train_ids.shape, train_ids.dtype)
print("valid tokens:", valid_ids.shape, valid_ids.dtype)
print("tokens por byte aprox:", round(len(train_ids) / max(1, train_txt.stat().st_size), 3))
```

#### Explicação linha a linha

A célula transforma os textos em um array NumPy 1-D de IDs — formato compacto e direto para sortear janelas de contexto.

- `def encode_texts(texts: list[str], tokenizer) -> np.ndarray:`: assinatura. `tokenizer` é "qualquer coisa com `.encode().ids`" (duck typing).
- `ids: list[int] = []`: buffer global de IDs.
- `eot = tokenizer.token_to_id(SPECIAL_TOKEN) if hasattr(tokenizer, "token_to_id") else None`: tenta descobrir o ID do token de fim de documento. Se o tokenizer não expõe `token_to_id`, deixa `eot = None` e simplesmente concatena os textos sem separador.
- `for text in texts:`: itera por documento.
- `ids.extend(tokenizer.encode(text).ids)`: adiciona os IDs do documento ao buffer global.
- `if eot is not None: ids.append(eot)`: separa documentos com `<|endoftext|>` — assim o modelo aprende que essas fronteiras existem.
- `dtype = np.uint16 if VOCAB_SIZE <= np.iinfo(np.uint16).max else np.int32`: escolhe o menor dtype que ainda comporta `VOCAB_SIZE`. `uint16` vai até 65.535. Para vocabs <=64k, `uint16` ocupa metade da memória de `int32` — relevante quando os arrays passam de centenas de milhões.
- `return np.asarray(ids, dtype=dtype)`: converte para array NumPy. `np.asarray` evita copiar se já for array.
- `train_ids = encode_texts(train_texts, tokenizer)`: tokeniza o split de treino.
- `valid_ids = encode_texts(valid_texts, tokenizer)`: idem para validação.
- `print("train tokens:", train_ids.shape, train_ids.dtype)`: imprime tamanho e dtype para sanidade.
- `print("valid tokens:", valid_ids.shape, valid_ids.dtype)`: idem.
- `print("tokens por byte aprox:", round(len(train_ids) / max(1, train_txt.stat().st_size), 3))`: razão tokens/byte. Em PT-BR, com BPE 4k, esperamos ~0.4-0.6 (cada byte vira ~0.5 tokens). `max(1, ...)` evita divisão por zero quando o arquivo é vazio.

```python
def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str = DEVICE) -> tuple[Tensor, Tensor]:
    assert dataset.ndim == 1
    assert len(dataset) > context_length + 1, "dataset pequeno demais para esse contexto"
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(dataset[i : i + context_length].astype(np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(dataset[i + 1 : i + context_length + 1].astype(np.int64)) for i in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


x, y = get_batch(train_ids, batch_size=4, context_length=16)
print(x.shape, y.shape)
print("x[0]:", x[0].tolist())
print("y[0]:", y[0].tolist())
```

#### Explicação linha a linha

`get_batch` é o que substitui um `DataLoader` em LMs causais: amostra janelas aleatórias do array de tokens e gera o par `(x, y)` deslocado em 1.

- `def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str = DEVICE) -> tuple[Tensor, Tensor]:`: assinatura. Note o type hint `tuple[Tensor, Tensor]` — devolvemos dois tensores.
- `assert dataset.ndim == 1`: o dataset deve ser 1-D (todos os tokens concatenados).
- `assert len(dataset) > context_length + 1, "dataset pequeno demais para esse contexto"`: precisamos de pelo menos `context_length + 1` tokens para formar uma janela com alvo. Mensagem de erro humana ajuda na hora do bug.
- `starts = torch.randint(0, len(dataset) - context_length, (batch_size,))`: amostra `batch_size` índices iniciais. O range é `[0, len-T)` — assim a janela `dataset[i:i+T]` cabe e ainda há um token em `i+T` para servir de target. `torch.randint` é exclusivo no limite superior.
- `x = torch.stack([torch.from_numpy(dataset[i : i + context_length].astype(np.int64)) for i in starts])`: para cada índice `i`, fatia `dataset[i:i+T]`, força `int64` (tipo exigido por `nn.Embedding`), converte para tensor e empilha. Resultado: shape `(batch_size, context_length)`.
- `y = torch.stack([torch.from_numpy(dataset[i + 1 : i + context_length + 1].astype(np.int64)) for i in starts])`: o target é a janela deslocada em 1. Mesmo shape de `x`.
- `return x.to(device, non_blocking=True), y.to(device, non_blocking=True)`: move para GPU. `non_blocking=True` permite cópia assíncrona quando a memória do host está em pinned memory — em CPU isso é ignorado.
- `x, y = get_batch(train_ids, batch_size=4, context_length=16)`: chamada de teste com batch pequeno.
- `print(x.shape, y.shape)`: deve imprimir `torch.Size([4, 16]) torch.Size([4, 16])`.
- `print("x[0]:", x[0].tolist())`: imprime a primeira sequência de IDs.
- `print("y[0]:", y[0].tolist())`: e seu target. Eyeball: `y[0]` deve ser exatamente `x[0]` deslocado em 1 (com um token novo no fim).

> **Pitfall comum**: o range correto é `randint(0, len(dataset) - context_length)`, não `len(dataset) - context_length - 1` nem `len(dataset)`. Se `i = len(dataset) - context_length - 1`, então `dataset[i + 1 : i + context_length + 1]` termina exatamente em `len(dataset)`, que é válido em slicing Python. Note que algumas implementações usam `len(dataset) - context_length - 1` por excesso de cautela; a versão acima é a do Karpathy/nanoGPT e é correta.

---

## 5. Primitivas: Linear e Embedding

`nn.Linear` e `nn.Embedding` parecem banais, mas são onde você começa a respeitar shapes.

Para uma Linear sem bias:

`out[..., d_out] = in[..., d_in] @ W[d_out, d_in].T`

O assignment passa pesos no formato `(d_out, d_in)`, como PyTorch armazena. Portanto a multiplicação correta é `x @ W.T`.

Embedding é indexação: para cada token ID, buscar a linha correspondente da matriz `(vocab_size, d_model)`. Não há one-hot explícito em produção porque seria desperdiçar memória e bandwidth.

```python
class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        std = math.sqrt(2 / (d_in + d_out))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


def run_linear(d_in: int, d_out: int, weights: Tensor, in_features: Tensor) -> Tensor:
    return in_features @ weights.T


def run_embedding(vocab_size: int, d_model: int, weights: Tensor, token_ids: Tensor) -> Tensor:
    return weights[token_ids]


test_x = torch.randn(2, 3, 5)
test_w = torch.randn(7, 5)
assert torch.allclose(run_linear(5, 7, test_w, test_x), F.linear(test_x, test_w))
print("Linear ok:", run_linear(5, 7, test_w, test_x).shape)
```

#### Explicação linha a linha

A célula define versões "manuais" de `nn.Linear` e `nn.Embedding` para fixar o entendimento de shapes e inicialização.

**Classe `Linear`**:
- `class Linear(nn.Module):`: herda de `nn.Module` para participar do sistema de parâmetros do PyTorch (autograd, `state_dict`, `to(device)` etc.).
- `def __init__(self, d_in: int, d_out: int):`: dimensões de entrada e saída.
- `super().__init__()`: chama o construtor da classe base — *obrigatório* em todo `nn.Module`.
- `self.weight = nn.Parameter(torch.empty(d_out, d_in))`: cria a matriz de pesos com shape `(d_out, d_in)` — convenção do PyTorch (`nn.Linear` armazena na mesma forma). `nn.Parameter` marca o tensor como treinável e o registra automaticamente em `parameters()`.
- `std = math.sqrt(2 / (d_in + d_out))`: desvio-padrão Xavier/Glorot — preserva a variância das ativações entre camadas.
- `nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)`: amostra de uma normal truncada (rejeita valores além de ±3σ). O `_` no final indica operação in-place.
- `def forward(self, x: Tensor) -> Tensor: return x @ self.weight.T`: produto matricial. Se `x` tem shape `(..., d_in)` e `weight` tem shape `(d_out, d_in)`, então `x @ weight.T` tem shape `(..., d_out)`. O `@` faz broadcasting nas dimensões prefixo.

**Classe `Embedding`**:
- `class Embedding(nn.Module):`: idem.
- `self.weight = nn.Parameter(torch.empty(vocab_size, d_model))`: tabela de busca: cada linha é o vetor de um token.
- `nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)`: inicialização N(0,1) truncada — convenção do GPT-2/Llama para embeddings.
- `def forward(self, token_ids: Tensor) -> Tensor: return self.weight[token_ids]`: indexação avançada do PyTorch. Se `token_ids` tem shape `(B, T)`, o resultado é `(B, T, d_model)` automaticamente. *Não* fazemos one-hot — seria O(V) memória por token sem necessidade.

**Funções `run_linear` / `run_embedding`**:
- `def run_linear(d_in, d_out, weights, in_features): return in_features @ weights.T`: versão "stateless" usada pelos testes do assignment, que passam pesos como argumento.
- `def run_embedding(vocab_size, d_model, weights, token_ids): return weights[token_ids]`: idem para Embedding.

**Bloco de teste**:
- `test_x = torch.randn(2, 3, 5)`: tensor 3-D com shape `(B=2, T=3, d_in=5)`.
- `test_w = torch.randn(7, 5)`: pesos `(d_out=7, d_in=5)`.
- `assert torch.allclose(run_linear(5, 7, test_w, test_x), F.linear(test_x, test_w))`: verifica equivalência com a implementação oficial do PyTorch. `F.linear(x, W)` faz `x @ W.T` internamente, e ambos devem dar o mesmo resultado bit-a-bit (atol default).
- `print("Linear ok:", run_linear(5, 7, test_w, test_x).shape)`: confirma shape `(2, 3, 7)`.

> **Intuição — por que Glorot/Xavier truncado?** A inicialização `std = sqrt(2 / (d_in + d_out))` é Xavier/Glorot, que mantém a variância do ativo aproximadamente constante entre camadas para ativações lineares ou simétricas (tanh). Para ReLU, a versão correta seria He (`std = sqrt(2 / d_in)`), que dobra a variância para compensar a metade morta da ReLU. Para SiLU/SwiGLU, a literatura usa Glorot por inércia — não há solução fechada limpa, e ablações em LLMs modernos mostram que a inicialização perde importância à medida que warmup, RMSNorm e residuals dominam. O assignment usa Glorot truncado porque o teste é determinístico contra esse esquema.

### Produção

Use `nn.Linear` e `nn.Embedding` em código real, salvo quando o assignment exige implementar. Eles ganham inicialização padronizada, integração com `state_dict`, compiladores, quantização, FSDP e ferramentas de inspeção. A versão manual serve para entender transposição, broadcasting de dimensões prefixo e custo de memória.

---

## 6. Funções numéricas estáveis: softmax e cross-entropy

Softmax ingênuo pode estourar: `exp(1000)` vira infinito. O truque padrão é subtrair o máximo por linha:

`softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`

Cross-entropy para logits é `-log softmax(logits)[target]`. Em produção, `F.cross_entropy` funde `log_softmax` e NLL de forma estável e eficiente. Implementar uma vez à mão ajuda a reconhecer quando não se deve aplicar softmax antes da loss.

> **Intuição — invariância de translação**: subtrair `max(x)` de todos os logits não muda a distribuição porque softmax é invariante a constantes aditivas: `softmax(x + c) = softmax(x)`. Isso é verdade matematicamente; em ponto flutuante, faz a diferença entre `inf/inf = NaN` e um cálculo limpo. A versão `logsumexp` (usada na cross-entropy abaixo) é a mesma ideia: `log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))`.

```python
def softmax(x: Tensor, dim: int = -1) -> Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    exps = torch.exp(shifted)
    return exps / exps.sum(dim=dim, keepdim=True)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -log_probs[torch.arange(targets.numel(), device=targets.device), targets].mean()


logits = torch.tensor([[1000.0, 1001.0, 999.0], [0.1, -0.2, 0.3]])
targets = torch.tensor([1, 2])
assert torch.allclose(softmax(logits, -1), F.softmax(logits, -1), atol=1e-6)
assert torch.allclose(cross_entropy(logits, targets), F.cross_entropy(logits, targets), atol=1e-6)
print("softmax/cross_entropy ok")
```

#### Explicação linha a linha

A célula implementa softmax e cross-entropy de forma numericamente estável.

**Função `softmax`**:
- `def softmax(x: Tensor, dim: int = -1) -> Tensor:`: assinatura. `dim=-1` é o último eixo — convenção para "vocabulário" no nosso caso.
- `shifted = x - x.max(dim=dim, keepdim=True).values`: o truque clássico. `x.max(dim, keepdim=True)` devolve um named tuple `(values, indices)`; pegamos `.values` e mantemos a dimensão (`keepdim=True`) para broadcasting. Subtrair o máximo torna o maior valor 0 e os outros ≤ 0 — `exp(0) = 1` e `exp(-grande) ≈ 0`, ambos seguros.
- `exps = torch.exp(shifted)`: exponencia.
- `return exps / exps.sum(dim=dim, keepdim=True)`: normaliza pela soma. `keepdim=True` mantém o shape para broadcasting limpo.

**Função `cross_entropy`**:
- `def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:`: assinatura. `logits` é `(N, V)`; `targets` é `(N,)` com IDs.
- `log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)`: calcula `log_softmax` direto via identidade `log_softmax(x) = x - logsumexp(x)`. `logsumexp` aplica o mesmo truque do max para evitar overflow (essa é uma op fundamental, fornecida pronta pelo PyTorch).
- `return -log_probs[torch.arange(targets.numel(), device=targets.device), targets].mean()`: indexa cada linha de `log_probs` no índice do target — `log_probs[0, targets[0]]`, `log_probs[1, targets[1]]`, etc. — usa `torch.arange(N)` do mesmo device dos targets para evitar bug de mismatch CPU/GPU. Negativa e tira a média sobre o batch (redução padrão `mean`).

**Bloco de teste**:
- `logits = torch.tensor([[1000.0, 1001.0, 999.0], [0.1, -0.2, 0.3]])`: caso "extremo" com logits gigantes para forçar o teste de estabilidade. Sem o shift do max, `exp(1000)` viraria `+inf` e o softmax seria `nan`.
- `targets = torch.tensor([1, 2])`: alvo: token 1 para a primeira linha, token 2 para a segunda.
- `assert torch.allclose(softmax(logits, -1), F.softmax(logits, -1), atol=1e-6)`: cruza com a implementação do PyTorch. Tolerância pequena porque `F.softmax` também é estável.
- `assert torch.allclose(cross_entropy(logits, targets), F.cross_entropy(logits, targets), atol=1e-6)`: idem para cross-entropy.
- `print("softmax/cross_entropy ok")`: log de sucesso.

> **Sanidade da loss inicial**: para um modelo com pesos aleatórios e vocab_size V, a cross-entropy esperada é `log(V)`, porque a distribuição predita é aproximadamente uniforme. Para V=4096, isso é ~8.32 nats; para V=32000, ~10.37 nats. Se você inicializa um modelo e a primeira loss está muito longe disso, há bug — provavelmente na inicialização ou na máscara causal. Esta é uma das melhores invariantes para depurar.

---

## 7. RMSNorm

LayerNorm centraliza e normaliza: subtrai média e divide pelo desvio. RMSNorm remove a centralização e normaliza apenas pela raiz da média dos quadrados:

`RMS(x) = sqrt(mean(x^2) + eps)`

`RMSNorm(x) = x / RMS(x) * weight`

Por que isso aparece em modelos modernos? Menos operações, boa estabilidade e desempenho empírico forte em arquiteturas decoder-only. Ele não substitui todo cuidado de otimização, mas reduz overhead de normalização.

> **Nota de 2026**: RMSNorm continua o padrão consensual em decoder-only LMs (Llama 3/4, Qwen 2/3, Mistral, DeepSeek V2/V3, GLM4.5, Kimi K2). Ablações de Zhang & Sennrich (2019) mostraram que a parte que LayerNorm "ganha" subtraindo a média é desprezível depois que o modelo treina; RMSNorm economiza ~15% em FLOPs de normalização e tem gradientes mais limpos. Há trabalhos recentes propondo alternativas — DyT (Dynamic Tanh), Derf (erf-based bounded normalizers), SoftCap — mas nenhum substituiu RMSNorm em produção até maio/2026. Cuidado importante: trabalhos de 2025-26 mostram acoplamento entre normalizador e otimizador (ex: Derf+Muon piora progressivamente vs RMSNorm+Muon), então mudar normalizador não é uma escolha modular como se assumia.

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        original_dtype = x.dtype
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * rms * self.weight).to(original_dtype)


def run_rmsnorm(d_model: int, eps: float, weights: Tensor, in_features: Tensor) -> Tensor:
    x = in_features.float()
    out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (out * weights).to(in_features.dtype)


z = torch.randn(2, 4, 8)
w = torch.ones(8)
print(run_rmsnorm(8, 1e-5, w, z).shape)
```

#### Explicação linha a linha

A célula implementa RMSNorm com upcast para float32 — o detalhe que diferencia uma implementação "didática" de uma "robusta em bf16".

**Classe `RMSNorm`**:
- `class RMSNorm(nn.Module):`: módulo treinável (tem `weight`).
- `def __init__(self, d_model: int, eps: float = 1e-5):`: aceita o `d_model` e um epsilon pequeno para evitar divisão por zero. `1e-5` é o default do Llama.
- `super().__init__()`: obrigatório.
- `self.eps = eps`: salva como atributo.
- `self.weight = nn.Parameter(torch.ones(d_model))`: vetor de ganho `γ` por canal, inicializado em 1. Começar em 1 significa "no início, RMSNorm é só normalização por RMS" — o ganho aprende correções depois.
- `def forward(self, x: Tensor) -> Tensor:`: forward com cuidado de dtype.
- `original_dtype = x.dtype`: lembra o dtype de entrada (provavelmente bf16 em treino).
- `x_float = x.float()`: faz upcast para fp32. Sem isso, `x.pow(2).mean()` em bf16 pode saturar quando `x` tem ativações grandes. Esta é uma das duas convenções (Llama, Mistral) — a outra é "manter bf16 e absorver o erro".
- `rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)`: calcula `1/sqrt(mean(x²) + eps)` em uma op só. `rsqrt` é mais rápido e numericamente melhor que `1/sqrt`.
- `return (x_float * rms * self.weight).to(original_dtype)`: aplica normalização e ganho, depois volta para o dtype original. O retorno tem o mesmo dtype da entrada — invisível para o resto do modelo.

**Função `run_rmsnorm`**:
- `def run_rmsnorm(d_model, eps, weights, in_features):`: versão "stateless" para os testes do assignment.
- `x = in_features.float()`: upcast.
- `out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)`: mesma fórmula.
- `return (out * weights).to(in_features.dtype)`: aplica ganho e devolve no dtype original.

**Bloco de teste**:
- `z = torch.randn(2, 4, 8)`: tensor `(B=2, T=4, d=8)`.
- `w = torch.ones(8)`: ganhos identidade.
- `print(run_rmsnorm(8, 1e-5, w, z).shape)`: confirma shape preservado `(2, 4, 8)`.

> **Pitfall — upcast para float32**: a normalização cuida de fazer tudo em float32 mesmo quando o modelo treina em bf16. Sem isso, `x.pow(2).mean()` em bf16 satura quando x tem ativações grandes. Ao final, devolvemos no dtype original. Esta é a convenção do Llama, e é por isso que o autocast de `torch.autocast` exclui RMSNorm/LayerNorm da lista de ops em low-precision por padrão.

---

## 8. SiLU e SwiGLU

O MLP do Transformer clássico usa algo como `Linear -> GeLU -> Linear`. Modelos estilo Llama usam uma família gated:

`SwiGLU(x) = W2( SiLU(W1 x) * W3 x )`

O produto elemento a elemento cria um portão contínuo. `W1` produz valores ativados, `W3` produz o caminho que será filtrado, e `W2` projeta de volta para `d_model`.

O detalhe de engenharia é que `d_ff` costuma ser arredondado para múltiplos favoráveis a GPU. Para estudo, use `d_ff = 4 * d_model` ou o valor do assignment. Para produção, alinhe a múltiplos como 64/128/256 dependendo do kernel/hardware.

> **Intuição — gating como modulação**: SwiGLU pode ser lido como "MLP que aprende quanto ativar". A coluna `SiLU(W1 x)` produz a ativação não-linear; a coluna `W3 x` é a "intensidade" que multiplica essa ativação. A rede descobre por canal quão forte cada feature deve disparar. Empiricamente (Shazeer 2020), isso supera GeLU/ReLU MLP a igual contagem de parâmetros — desde que você ajuste `d_ff` para 8/3 × d_model em vez de 4 × d_model, mantendo o número de parâmetros constante (3 matrizes em vez de 2). Este "8/3" é o que você vê em Llama (ex: d_ff=11008 para d_model=4096, que é ~2.69× = 8/3 × 1.008).

> **Nota de 2026**: SwiGLU continua dominante em decoder-only LMs frontier. Variantes minoritárias como GeGLU (com GeLU em vez de SiLU) e ReGLU (com ReLU) aparecem em modelos específicos, mas SwiGLU é o padrão de fato em 2026 — Llama 4, Qwen 3, DeepSeek V3, Kimi K2, GLM4.5 todos usam.

```python
def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def run_swiglu(d_model: int, d_ff: int, w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor, in_features: Tensor) -> Tensor:
    hidden = F.silu(in_features @ w1_weight.T) * (in_features @ w3_weight.T)
    return hidden @ w2_weight.T


x_small = torch.randn(2, 3, 16)
print(SwiGLU(16, 64)(x_small).shape)
```

#### Explicação linha a linha

A célula implementa SwiGLU — o "MLP gated" usado em LLaMA, Mistral, Qwen et al.

**Função `silu`**:
- `def silu(x: Tensor) -> Tensor: return x * torch.sigmoid(x)`: SiLU (também chamada Swish) é `x * σ(x)`. É lisa, não monotônica perto de zero, e empiricamente bate ReLU/GeLU em transformers grandes. PyTorch já tem `F.silu`, mas implementar à mão é trivial.

**Classe `SwiGLU`**:
- `class SwiGLU(nn.Module):`: módulo gated.
- `def __init__(self, d_model: int, d_ff: int):`: dimensões de modelo e do hidden interno.
- `super().__init__()`: obrigatório.
- `self.w1 = Linear(d_model, d_ff)`: projeção "gate" (vai para a SiLU).
- `self.w2 = Linear(d_ff, d_model)`: projeção de saída.
- `self.w3 = Linear(d_model, d_ff)`: projeção "value" (vai para o produto elemento a elemento).
- `def forward(self, x: Tensor) -> Tensor: return self.w2(silu(self.w1(x)) * self.w3(x))`: o cerne. Lê: "ative `W1 x` com SiLU, multiplique elemento a elemento por `W3 x`, depois projete via `W2`". Note que `*` aqui é multiplicação elemento a elemento (Hadamard) — exatamente o portão.

**Função `run_swiglu`**:
- `def run_swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features):`: versão stateless.
- `hidden = F.silu(in_features @ w1_weight.T) * (in_features @ w3_weight.T)`: aplica W1 (com SiLU) e W3 (linear puro), multiplica.
- `return hidden @ w2_weight.T`: projeção final.

**Bloco de teste**:
- `x_small = torch.randn(2, 3, 16)`: tensor de teste shape `(2, 3, 16)` — `d_model=16`.
- `print(SwiGLU(16, 64)(x_small).shape)`: instancia SwiGLU com `d_ff=64` (4×) e roda. Espera-se `(2, 3, 16)` na saída — d_model preservado.

---

## 9. RoPE: posições como rotações

Atenção por si só é permutacional: se você embaralhar tokens e não der posição, o modelo não sabe a ordem. RoPE injeta posição rotacionando pares de dimensões de queries e keys. Para cada posição `p` e frequência `theta_i`, aplicamos:

`[x_even, x_odd] -> [x_even cos(p theta_i) - x_odd sin(p theta_i), x_even sin(p theta_i) + x_odd cos(p theta_i)]`

Por que RoPE é atraente:

- não soma uma tabela posicional separada aos embeddings;
- preserva uma forma de informação relativa entre posições via produto interno entre Q e K rotacionados;
- extrapola melhor que embeddings posicionais aprendidos, embora extrapolação longa ainda exija cuidado.

No assignment, RoPE é aplicado em Q e K por cabeça, com dimensão `d_model // num_heads`.

> **Intuição — relatividade emergente**: a propriedade chave de RoPE é que o produto interno `<Rotate(q, m), Rotate(k, n)>` depende apenas de `m - n` (a distância), não de `m` e `n` absolutos. Isso significa que o mecanismo de atenção naturalmente vê *posições relativas*, sem nenhuma tabela explícita disso. É elegante: posição absoluta entra na rotação, posição relativa emerge no produto interno.

> **Nota de 2026 — context extension via RoPE**: o θ base de 10.000 do RoPE original é uma escolha que satura em ~4-8k tokens. O ecossistema desenvolveu várias técnicas para estender contexto:
>
> - **Position Interpolation (PI)**: Chen et al. 2023 — escala todas as frequências por `1/s`. Funciona até s≈8.
> - **NTK-aware scaling**: ajusta a base `θ` em vez de escalar uniformemente; preserva alta frequência (informação local).
> - **YaRN** (Peng et al. 2023, atualizado em fev/2026): combina NTK-by-parts (3 grupos de frequência tratados diferente) com *attention temperature scaling*. Estado da arte para extensão até 128k+ com fine-tuning leve. Llama, Qwen, DeepSeek, Mistral, gpt-oss usam YaRN para extensão.
> - **LongRoPE**: extensão até 2M+ tokens via busca não-uniforme.
>
> Llama 3.1 e modelos posteriores frequentemente nascem com `θ=500.000` em vez de 10.000, justamente para suportar contexto longo nativamente sem precisar de YaRN. Para o assignment, mantenha 10k — é o que os testes esperam.

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10_000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE exige dimensao par"
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        # x: (..., seq, dim)
        seq_len = x.shape[-2]
        if token_positions is None:
            cos = self.cos[:seq_len].to(device=x.device, dtype=x.dtype)
            sin = self.sin[:seq_len].to(device=x.device, dtype=x.dtype)
        else:
            cos = self.cos[token_positions].to(device=x.device, dtype=x.dtype)
            sin = self.sin[token_positions].to(device=x.device, dtype=x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        while cos.ndim < x_even.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


def run_rope(d_k: int, theta: float, max_seq_len: int, in_query_or_key: Tensor, token_positions: Tensor) -> Tensor:
    rope = RotaryEmbedding(d_k, max_seq_len=max_seq_len, theta=theta).to(in_query_or_key.device)
    return rope(in_query_or_key, token_positions)


rope_test = torch.randn(2, 5, 8)
pos = torch.arange(5).unsqueeze(0).expand(2, -1)
print(run_rope(8, 10_000.0, 5, rope_test, pos).shape)
```

#### Explicação linha a linha

RoPE injeta posição rotacionando pares de canais de Q e K. Esta célula implementa a versão *even/odd interleaved* (a do CS336).

**Classe `RotaryEmbedding`**:
- `class RotaryEmbedding(nn.Module):`: módulo *sem parâmetros treináveis* (cos/sin pré-calculados).
- `def __init__(self, dim: int, max_seq_len: int, theta: float = 10_000.0):`: aceita a dimensão da rotação (=`head_dim`), o comprimento máximo de sequência e a base θ (10k canônico).
- `assert dim % 2 == 0, "RoPE exige dimensao par"`: precisamos pares de canais para rotacionar.
- `inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))`: frequências inversas. Para `dim=8` e θ=10000: `arange(0,8,2) = [0, 2, 4, 6]`; dividido por 8 dá `[0, 0.25, 0.5, 0.75]`; θ elevado a isso dá `[1, 10, 100, 1000]`; o recíproco é `[1, 0.1, 0.01, 0.001]`. Cada par de canais oscila em uma frequência diferente — pares baixos giram devagar (capturam contexto longo), pares altos giram rápido (capturam local).
- `positions = torch.arange(max_seq_len).float()`: vetor `[0, 1, 2, ..., max_seq_len-1]`.
- `freqs = torch.outer(positions, inv_freq)`: produto externo. Resultado shape `(max_seq_len, dim/2)` — entrada `[p, i]` é `p * inv_freq[i]`.
- `self.register_buffer("cos", torch.cos(freqs), persistent=False)`: `cos` da matriz; `persistent=False` evita que ele entre no `state_dict` (é determinístico, não precisa salvar). `register_buffer` faz o tensor seguir o módulo em `.to(device)` automaticamente.
- `self.register_buffer("sin", torch.sin(freqs), persistent=False)`: idem para `sin`.
- `def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:`: aceita `x` com shape `(..., seq, dim)` e opcionalmente um tensor de posições explícito (útil para KV-cache).
- `seq_len = x.shape[-2]`: extrai o comprimento da sequência.
- `if token_positions is None: cos = self.cos[:seq_len].to(...); sin = self.sin[:seq_len].to(...)`: caso default, pega cos/sin para posições `[0, seq_len)`. `.to(device=x.device, dtype=x.dtype)` garante mesmo device/dtype para a multiplicação.
- `else: cos = self.cos[token_positions].to(...); sin = self.sin[token_positions].to(...)`: indexa pelas posições explícitas, permitindo posições não contíguas (geração com cache).
- `x_even = x[..., 0::2]`: canais pares (índices 0, 2, 4, ...).
- `x_odd = x[..., 1::2]`: canais ímpares (índices 1, 3, 5, ...).
- `while cos.ndim < x_even.ndim: cos = cos.unsqueeze(-3); sin = sin.unsqueeze(-3)`: insere dimensões 1 antes de `seq` em cos/sin para casar com `(batch, heads, seq, dim/2)`. Cada `unsqueeze(-3)` insere um eixo de tamanho 1 — broadcasting faz o resto.
- `out_even = x_even * cos - x_odd * sin`: parte real da rotação 2D. Equivalente a `x_even cos(θ) - x_odd sin(θ)` para o ângulo θ = posição × inv_freq.
- `out_odd = x_even * sin + x_odd * cos`: parte imaginária.
- `return torch.stack((out_even, out_odd), dim=-1).flatten(-2)`: empilha os dois canais "even" e "odd" rotacionados em uma nova dimensão final — shape `(..., dim/2, 2)` — depois `flatten(-2)` colapsa as duas últimas em uma única de tamanho `dim`. Isso restabelece o interleaving original `[e0, o0, e1, o1, ...]`.

**Função `run_rope`**:
- `def run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions):`: versão stateless dos testes.
- `rope = RotaryEmbedding(d_k, max_seq_len=max_seq_len, theta=theta).to(in_query_or_key.device)`: instancia e move para device.
- `return rope(in_query_or_key, token_positions)`: aplica.

**Bloco de teste**:
- `rope_test = torch.randn(2, 5, 8)`: tensor `(B=2, T=5, dim=8)`.
- `pos = torch.arange(5).unsqueeze(0).expand(2, -1)`: vetor `[0,1,2,3,4]`, expandido para `(2, 5)` — mesma posição em ambos os elementos do batch. `expand` não copia memória, só ajusta strides.
- `print(run_rope(8, 10_000.0, 5, rope_test, pos).shape)`: confirma shape preservado `(2, 5, 8)`.

> **Pitfall — convenção even/odd vs metade-metade**: existem duas convenções para "pares de dimensões" em RoPE. Esta versão usa even/odd interleaved (`x[0::2]` e `x[1::2]`); o Llama original usa "metade-metade" (`x[:d/2]` e `x[d/2:]`). As duas são matematicamente equivalentes a uma permutação de canais, mas pesos não são portáveis entre as duas convenções. O assignment do CS336 usa even/odd; HuggingFace Transformers expõe as duas via `rope_type`.

---

## 10. Atenção causal escalada

Atenção calcula similaridade entre queries e keys:

`scores = Q K^T / sqrt(d_k)`

Depois mascara o futuro, aplica softmax e combina values:

`out = softmax(mask(scores)) V`

O fator `sqrt(d_k)` evita que o produto interno cresça com a dimensão e empurre softmax para saturação. A máscara causal impede que a posição `t` veja tokens `> t`, preservando o objetivo autoregressivo.

> **Intuição — por que `sqrt(d_k)`?** Se `q` e `k` têm componentes i.i.d. com variância 1, então `q · k` tem variância `d_k`. Sem dividir por `sqrt(d_k)`, scores crescem com a dimensão e softmax fica perto de one-hot — o gradiente desaparece. Dividir por `sqrt(d_k)` mantém a variância de scores em ~1 independente de d_k. Esta é a razão original de Vaswani et al. (2017); QK-norm e técnicas posteriores tratam casos onde isso ainda não basta.

```python
def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = F.softmax(scores, dim=-1)
    return probs @ V


def causal_mask(seq_len: int, device: str | torch.device) -> Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


q = torch.randn(2, 4, 8)
k = torch.randn(2, 4, 8)
v = torch.randn(2, 4, 8)
m = causal_mask(4, q.device)
print(scaled_dot_product_attention(q, k, v, m).shape)
```

#### Explicação linha a linha

A célula implementa atenção escalada com máscara de forma transparente — boa para entender a fórmula, ruim para produção (não usa FlashAttention).

**Função `scaled_dot_product_attention`**:
- `def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:`: aceita Q, K, V já com shape `(..., seq, d_k)` e máscara booleana opcional.
- `d_k = Q.shape[-1]`: dimensão por cabeça — última do tensor.
- `scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)`: produto interno entre todas as queries e todas as keys, normalizado por √dₖ. `K.transpose(-2, -1)` troca os dois últimos eixos — equivalente a `K^T` em cada batch. Resultado: shape `(..., seq_q, seq_k)`.
- `if mask is not None:`: máscara opcional.
- `scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)`: posições onde `mask` é `False` viram um número *muito* negativo (`-finfo.min` para o dtype atual; em fp32 ≈ -3.4e38). Após softmax, esses pontos viram ~0. `~mask` é negação booleana.
- `probs = F.softmax(scores, dim=-1)`: softmax sobre o eixo dos keys, transformando scores em probabilidades.
- `return probs @ V`: combina linearmente values pelo peso das probs. Shape final: `(..., seq_q, d_v)`.

**Função `causal_mask`**:
- `def causal_mask(seq_len: int, device: str | torch.device) -> Tensor:`: cria máscara causal triangular inferior.
- `return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))`: matriz `T×T` de booleanos. `tril` zera o triângulo superior — `True` significa "essa posição é visível". Ex.: para `T=4`, devolve `[[T,F,F,F],[T,T,F,F],[T,T,T,F],[T,T,T,T]]`.

**Bloco de teste**:
- `q = torch.randn(2, 4, 8)`: shape `(B=2, T=4, d_k=8)`.
- `k = torch.randn(2, 4, 8)`: idem.
- `v = torch.randn(2, 4, 8)`: idem.
- `m = causal_mask(4, q.device)`: máscara `(4, 4)`.
- `print(scaled_dot_product_attention(q, k, v, m).shape)`: imprime shape `(2, 4, 8)` — d_k preservado, T preservado.

### Produção: `F.scaled_dot_product_attention`

Desde PyTorch 2.x, `torch.nn.functional.scaled_dot_product_attention` escolhe kernels otimizados quando possível, incluindo caminhos tipo FlashAttention em GPU compatível. Em produção, prefira essa função. A versão manual é para entender mascaramento, shapes e estabilidade.

> **Nota de 2026 — paisagem de kernels de atenção**:
>
> - **FlashAttention 3** (Shah et al. 2024, padrão H100/B200 desde 2025): explora assincronia entre Tensor Cores e TMA, sobreposição warp-specialization, e FP8 nativo. Em H100 atinge 1.5–2× sobre FA2 (até 740 TFLOPs em FP16, ~1.2 PFLOPs em FP8). FA3 é o caminho default em PyTorch 2.7+ via `F.scaled_dot_product_attention`.
> - **FlexAttention** (PyTorch 2.5+): API que permite definir máscaras e biases custom (sliding window, document mask, ALiBi, soft-cap) e gerar kernel via `torch.compile`. Em PyTorch 2.7 ganhou suporte CPU. Útil para arquiteturas que escapam de "atenção causal pura".
> - **FlashInfer**: kernels para inferência (paged KV-cache, prefill chunked, speculative decoding).
>
> Para o assignment, `F.scaled_dot_product_attention(Q, K, V, is_causal=True)` é o caminho — PyTorch escolhe FA3, FA2 ou math fallback automaticamente conforme hardware e dtype.

```python
def production_sdpa(q: Tensor, k: Tensor, v: Tensor, is_causal: bool = True) -> Tensor:
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal)
```

#### Explicação linha a linha

Wrapper de uma linha sobre o SDPA do PyTorch — é o que você deve usar em produção.

- `def production_sdpa(q, k, v, is_causal=True):`: assinatura limpa, default causal.
- `F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal)`: chama a função fundida do PyTorch. Internamente, ela despacha para FlashAttention 2/3 em GPUs Hopper/Ampere com bf16/fp16, ou para o caminho "math" (igual à versão manual) em CPU/casos excepcionais. `attn_mask=None` + `is_causal=True` é a forma idiomática para atenção causal pura — é mais rápido que passar a máscara triangular explicitamente, porque o kernel sabe pular o triângulo superior sem materializá-lo.

---

## 11. Multi-head self-attention

Uma única cabeça comprime tudo em um espaço de atenção. Múltiplas cabeças permitem subespaços diferentes: sintaxe, pontuação, dependências locais, nomes próprios, padrões estilísticos etc. Não interprete cabeças de forma mística, mas a decomposição aumenta capacidade mantendo matmuls grandes e eficientes.

Shapes usados abaixo:

- entrada `x`: `(batch, seq, d_model)`
- Q/K/V projetados: `(batch, seq, d_model)`
- separados por cabeça: `(batch, heads, seq, head_dim)`
- saída concatenada: `(batch, seq, d_model)`

> **Nota de 2026 — MHA, MQA, GQA, MLA**: em modelos modernos, MHA "puro" foi quase totalmente substituído por variantes que reduzem o KV-cache (o gargalo de inferência long-context). A taxonomia em maio/2026:
>
> - **MHA** (Vaswani 2017): cada cabeça tem suas próprias K, V. KV-cache: `n_heads × head_dim × T`.
> - **MQA** (Shazeer 2019): uma única K, V compartilhada por todas as cabeças. Cache reduzido n_heads vezes, mas perde qualidade.
> - **GQA** (Ainslie 2023): grupos de cabeças que compartilham K, V. Compromisso usado em Llama 2/3/4, Qwen, Mistral. Para n_kv_heads=8 em modelo com 64 query heads: cache 8× menor que MHA, qualidade ~igual.
> - **MLA — Multi-Head Latent Attention** (DeepSeek V2/V3, 2024-25): comprime K e V para um vetor latente de baixa dimensão antes de cachear; reconstrói via projeções up. DeepSeek V2 reduziu KV-cache em 93.3% comparado ao seu MHA equivalente, e em ablações *superou* MHA. Adotado em Kimi K2 (1T params), GLM4.5, e várias migrações pós-treino (TransMLA mostra que GQA pode ser convertido em MLA equivalente).
>
> Para o assignment continue com MHA — é a base canônica e os testes assumem n_kv_heads = n_q_heads. Em produção 2026, a escolha "default" para um novo modelo decoder-only pequeno-médio (1B-100B) é GQA com `n_kv_heads = n_q_heads / 4` ou MLA. Para o seu próprio próximo experimento depois do CS336, vale conhecer pelo menos GQA.

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, rope_theta: float = 10_000.0, use_rope: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, theta=rope_theta) if use_rope else None

    def _split_heads(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        return self.output_proj(self._merge_heads(attn))


def run_multihead_self_attention(d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features):
    b, t, _ = in_features.shape
    head_dim = d_model // num_heads
    q = (in_features @ q_proj_weight.T).view(b, t, num_heads, head_dim).transpose(1, 2)
    k = (in_features @ k_proj_weight.T).view(b, t, num_heads, head_dim).transpose(1, 2)
    v = (in_features @ v_proj_weight.T).view(b, t, num_heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = out.transpose(1, 2).contiguous().view(b, t, d_model)
    return out @ o_proj_weight.T
```

#### Explicação linha a linha

A célula define `MultiHeadSelfAttention` — combina projeções Q/K/V, RoPE opcional e SDPA fundido.

**Classe `MultiHeadSelfAttention`**:
- `def __init__(self, d_model, num_heads, max_seq_len, rope_theta=10_000.0, use_rope=True):`: hiperparâmetros do módulo.
- `assert d_model % num_heads == 0`: head_dim precisa ser inteiro.
- Atribuições `self.d_model`, `self.num_heads`, `self.head_dim`, `self.use_rope`: salvam configs como atributos.
- `self.head_dim = d_model // num_heads`: dimensão por cabeça.
- `self.q_proj = Linear(d_model, d_model)`: projeção `W_Q`. Note que mantemos `d_model` (não `head_dim × num_heads` separado por cabeça); o split em cabeças é feito por `view`/`transpose`.
- `self.k_proj = Linear(d_model, d_model)`: idem para K.
- `self.v_proj = Linear(d_model, d_model)`: idem para V.
- `self.output_proj = Linear(d_model, d_model)`: projeção final `W_O` que mistura cabeças.
- `self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, theta=rope_theta) if use_rope else None`: RoPE atua *por cabeça* — daí `head_dim`.

**Helper `_split_heads`**:
- `def _split_heads(self, x: Tensor) -> Tensor:`: transforma `(B, T, d_model)` em `(B, num_heads, T, head_dim)`.
- `b, t, d = x.shape`: extrai shape.
- `return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)`: primeiro reorganiza em `(B, T, H, d_h)`, depois troca dims 1 e 2 para `(B, H, T, d_h)`. Esse layout é o que SDPA espera.

**Helper `_merge_heads`**:
- `def _merge_heads(self, x: Tensor) -> Tensor:`: faz o inverso.
- `b, h, t, d = x.shape`: extrai shape.
- `return x.transpose(1, 2).contiguous().view(b, t, h * d)`: transpõe para `(B, T, H, d_h)`, força contiguidade (porque `view` exige memória contígua), depois colapsa as duas últimas dims em `d_model`.

**Forward**:
- `q = self._split_heads(self.q_proj(x))`: projeta e split. Shape: `(B, H, T, d_h)`.
- `k = self._split_heads(self.k_proj(x))`: idem para K.
- `v = self._split_heads(self.v_proj(x))`: idem para V.
- `if self.rope is not None: q = self.rope(q, token_positions); k = self.rope(k, token_positions)`: aplica RoPE em Q e K. *Não* aplica em V — RoPE é só para o cálculo de similaridade.
- `attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)`: atenção fundida.
- `return self.output_proj(self._merge_heads(attn))`: junta cabeças e projeta.

**Função stateless `run_multihead_self_attention`** (versão sem RoPE para o teste do assignment):
- `b, t, _ = in_features.shape`: extrai batch e seq.
- `head_dim = d_model // num_heads`: cabeças.
- `q = (in_features @ q_proj_weight.T).view(b, t, num_heads, head_dim).transpose(1, 2)`: linhão que faz "projeção + split heads" em uma expressão.
- `k`, `v`: idem.
- `out = F.scaled_dot_product_attention(q, k, v, is_causal=True)`: atenção.
- `out = out.transpose(1, 2).contiguous().view(b, t, d_model)`: merge.
- `return out @ o_proj_weight.T`: projeção de saída.

> **Pitfall — `.contiguous()` antes do `.view()`**: depois de `transpose`, o tensor tem strides não contíguos. `view` exige contiguidade. Esquecer isso lança `RuntimeError: view size is not compatible with input tensor's size and stride`. A alternativa é usar `reshape`, que faz a cópia se necessário, mas `contiguous()` + `view` é mais explícito sobre quando o custo de cópia ocorre.

```python
def run_multihead_self_attention_with_rope(
    d_model, num_heads, max_seq_len, theta,
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
    in_features, token_positions=None,
):
    b, t, _ = in_features.shape
    head_dim = d_model // num_heads
    q = (in_features @ q_proj_weight.T).view(b, t, num_heads, head_dim).transpose(1, 2)
    k = (in_features @ k_proj_weight.T).view(b, t, num_heads, head_dim).transpose(1, 2)
    v = (in_features @ v_proj_weight.T).view(b, t, num_heads, head_dim).transpose(1, 2)
    rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, theta=theta).to(in_features.device)
    q = rope(q, token_positions)
    k = rope(k, token_positions)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = out.transpose(1, 2).contiguous().view(b, t, d_model)
    return out @ o_proj_weight.T


attn = MultiHeadSelfAttention(d_model=64, num_heads=4, max_seq_len=128)
print(attn(torch.randn(2, 16, 64)).shape)
```

#### Explicação linha a linha

Variante stateless do MHA *com* RoPE — exatamente o que o teste do assignment exige.

**Função `run_multihead_self_attention_with_rope`**:
- Assinatura recebe pesos das 4 projeções (`q_proj_weight`, `k_proj_weight`, `v_proj_weight`, `o_proj_weight`) e parâmetros de RoPE (`max_seq_len`, `theta`).
- `b, t, _ = in_features.shape`: extrai batch e seq.
- `head_dim = d_model // num_heads`: cabeças.
- `q = (in_features @ q_proj_weight.T).view(b, t, num_heads, head_dim).transpose(1, 2)`: projeção `W_Q`, split em cabeças, transpose para `(B, H, T, d_h)`.
- `k`: idem.
- `v`: idem.
- `rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, theta=theta).to(in_features.device)`: instancia RoPE *fresh* a cada chamada — caro em loop, mas o teste roda uma vez. Em produção a `RoPE` vive como atributo do módulo.
- `q = rope(q, token_positions)`: rotaciona Q.
- `k = rope(k, token_positions)`: rotaciona K.
- `out = F.scaled_dot_product_attention(q, k, v, is_causal=True)`: atenção fundida.
- `out = out.transpose(1, 2).contiguous().view(b, t, d_model)`: merge heads.
- `return out @ o_proj_weight.T`: projeção de saída.

**Bloco de teste**:
- `attn = MultiHeadSelfAttention(d_model=64, num_heads=4, max_seq_len=128)`: 4 cabeças × 16 dims = 64.
- `print(attn(torch.randn(2, 16, 64)).shape)`: imprime `(2, 16, 64)` — preserva shape.

---

## 12. Bloco Transformer pre-norm

O bloco usado aqui é decoder-only:

`x = x + Attention(RMSNorm(x))`

`x = x + SwiGLU(RMSNorm(x))`

Esse arranjo é pre-norm: normalizamos antes de cada subcamada. Ele tende a treinar de forma mais estável em profundidade do que post-norm clássico. Os caminhos residuais são cruciais: eles preservam um caminho de gradiente e permitem que cada bloco aprenda uma correção sobre a representação atual.

> **Intuição — por que pre-norm > post-norm em profundidade**: em post-norm (`LN(x + Sublayer(x))`), a norma do residual é "limpa" a cada bloco; em pre-norm (`x + Sublayer(LN(x))`), a norma do residual cresce com a profundidade. Isso parece ruim, mas significa que o gradiente do output flui *direto* até qualquer bloco intermediário sem atenuação por nada não-linear. Resultado: warmup pode ser muito mais agressivo e o modelo treina estável até centenas de camadas. Praticamente todos os decoder-only LMs modernos são pre-norm. Variantes como "sandwich norm" (norm antes E depois da sublayer) aparecem em casos específicos para controlar explosões em profundidade extrema (>100 camadas).

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, rope_theta: float):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, rope_theta, use_rope=True)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


def run_transformer_block(d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features):
    x = in_features
    norm1 = run_rmsnorm(d_model, 1e-5, weights["ln1.weight"], x)
    attn_out = run_multihead_self_attention_with_rope(
        d_model, num_heads, max_seq_len, theta,
        weights["attn.q_proj.weight"], weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"], weights["attn.output_proj.weight"],
        norm1,
    )
    x = x + attn_out
    norm2 = run_rmsnorm(d_model, 1e-5, weights["ln2.weight"], x)
    ffn_out = run_swiglu(d_model, d_ff, weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"], norm2)
    return x + ffn_out
```

#### Explicação linha a linha

Esta célula define o `TransformerBlock` — duas sub-camadas (atenção + FFN), cada uma com pre-norm e residual.

**Classe `TransformerBlock`**:
- `class TransformerBlock(nn.Module):`: módulo composto.
- `def __init__(self, d_model, num_heads, d_ff, max_seq_len, rope_theta):`: hiperparâmetros do bloco.
- `super().__init__()`: obrigatório.
- `self.ln1 = RMSNorm(d_model)`: norm antes da atenção.
- `self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, rope_theta, use_rope=True)`: módulo de atenção.
- `self.ln2 = RMSNorm(d_model)`: norm antes do FFN.
- `self.ffn = SwiGLU(d_model, d_ff)`: MLP gated.
- `def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:`: forward.
- `x = x + self.attn(self.ln1(x), token_positions=token_positions)`: pre-norm para a entrada da atenção, depois soma residual. O `+` é a "estrada principal" do gradiente.
- `x = x + self.ffn(self.ln2(x))`: pre-norm + FFN + residual.
- `return x`: devolve o estado atualizado.

**Função stateless `run_transformer_block`** (para teste do assignment):
- `x = in_features`: alias.
- `norm1 = run_rmsnorm(d_model, 1e-5, weights["ln1.weight"], x)`: aplica primeira RMSNorm com pesos do dicionário.
- `attn_out = run_multihead_self_attention_with_rope(...)`: chama a versão stateless de MHA com RoPE, passando os pesos certos do `state_dict`.
- `x = x + attn_out`: residual da atenção.
- `norm2 = run_rmsnorm(d_model, 1e-5, weights["ln2.weight"], x)`: segunda RMSNorm.
- `ffn_out = run_swiglu(d_model, d_ff, weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"], norm2)`: SwiGLU stateless.
- `return x + ffn_out`: residual do FFN.

A parte mais sutil é o uso do `state_dict` — chaves como `"ln1.weight"`, `"attn.q_proj.weight"` etc. seguem a convenção do PyTorch (`<nome_do_módulo>.<atributo>`). Isso permite rodar a função carregando pesos de um teste sem precisar instanciar a classe.

---

## 13. Transformer LM completo

O language model causal combina:

- embedding de tokens;
- pilha de blocos Transformer;
- RMSNorm final;
- cabeça linear para logits de vocabulário.

Cada posição produz logits para o próximo token. Durante treino, calculamos cross-entropy sobre todas as posições do batch. Durante geração, usamos apenas os logits da última posição.

Nota sobre weight tying: muitos modelos usam a mesma matriz para embedding e `lm_head`. O assignment pode esperar pesos separados em testes. A classe abaixo permite ligar ou desligar.

> **Intuição — weight tying**: amarrar `lm_head.weight = token_embeddings.weight` reduz parâmetros em `vocab_size × d_model` (ex: 32k × 4096 = 128M parâmetros economizados em Llama 3). A motivação semântica é que tanto embedding quanto lm_head são "mapeamentos token ↔ vetor"; tying força essa simetria. Em modelos pequenos, tying ajuda; em modelos grandes (>20B), separar é o padrão atual (Llama 3, Qwen 3, DeepSeek) — a explicação aceita é que com vocab_size grande, lm_head e embedding querem geometrias ligeiramente diferentes, e parâmetros separados permitem isso.

```python
@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int = 256
    d_model: int = 384
    num_layers: int = 6
    num_heads: int = 6
    d_ff: int = 1536
    rope_theta: float = 10_000.0
    tie_weights: bool = False


class TransformerLM(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.d_ff, config.context_length, config.rope_theta)
            for _ in range(config.num_layers)
        ])
        self.ln_final = RMSNorm(config.d_model)
        self.lm_head = Linear(config.d_model, config.vocab_size)
        if config.tie_weights:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, idx: Tensor) -> Tensor:
        _, t = idx.shape
        assert t <= self.config.context_length
        x = self.token_embeddings(idx)
        positions = torch.arange(t, device=idx.device).unsqueeze(0).expand(idx.shape[0], -1)
        for layer in self.layers:
            x = layer(x, token_positions=positions)
        x = self.ln_final(x)
        return self.lm_head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


tiny_cfg = GPTConfig(vocab_size=VOCAB_SIZE, context_length=64, d_model=128, num_layers=2, num_heads=4, d_ff=512)
tiny_model = TransformerLM(tiny_cfg).to(DEVICE)
print("params:", count_parameters(tiny_model))
print(tiny_model(torch.randint(0, VOCAB_SIZE, (2, 16), device=DEVICE)).shape)
```

#### Explicação linha a linha

Esta célula define `GPTConfig` (dataclass de configuração) e `TransformerLM` (modelo completo).

**Dataclass `GPTConfig`**:
- `@dataclass`: decorador que gera `__init__`, `__repr__`, `__eq__`.
- `vocab_size: int`: campo obrigatório (sem default).
- `context_length: int = 256`: comprimento máximo do contexto.
- `d_model: int = 384`: dimensão do modelo.
- `num_layers: int = 6`: profundidade.
- `num_heads: int = 6`: cabeças.
- `d_ff: int = 1536`: dimensão do FFN. Nota: 1536/384 = 4 (4× clássico, não 8/3 do Llama). É uma escolha didática; em produção, alinhe a múltiplos de 128/256.
- `rope_theta: float = 10_000.0`: base de RoPE.
- `tie_weights: bool = False`: flag para weight tying.

**Classe `TransformerLM`**:
- `def __init__(self, config: GPTConfig):`: construtor recebe a config inteira (clean — uma única fonte de verdade).
- `super().__init__()`: obrigatório.
- `self.config = config`: salva a config como atributo (útil para `forward` e debugging).
- `self.token_embeddings = Embedding(config.vocab_size, config.d_model)`: tabela de embedding.
- `self.layers = nn.ModuleList([TransformerBlock(...) for _ in range(config.num_layers)])`: lista de blocos. `nn.ModuleList` (não `list`) registra cada bloco como sub-módulo, garantindo `parameters()`, `to(device)` e `state_dict` corretos.
- `self.ln_final = RMSNorm(config.d_model)`: norm final (canônico em pre-norm).
- `self.lm_head = Linear(config.d_model, config.vocab_size)`: projeção para logits.
- `if config.tie_weights: self.lm_head.weight = self.token_embeddings.weight`: weight tying — *o mesmo* tensor é parâmetro de ambos. Não é cópia: gradientes acumulam nas duas referências.
- `def forward(self, idx: Tensor) -> Tensor:`: `idx` shape `(B, T)` com IDs.
- `_, t = idx.shape`: descarta `B`, pega `T`.
- `assert t <= self.config.context_length`: protege contra overflow de RoPE.
- `x = self.token_embeddings(idx)`: lookup. Shape: `(B, T, d_model)`.
- `positions = torch.arange(t, device=idx.device).unsqueeze(0).expand(idx.shape[0], -1)`: monta um tensor de posições absolutas `(B, T)` com `[0,1,...,T-1]` em cada linha. `expand` evita cópia.
- `for layer in self.layers: x = layer(x, token_positions=positions)`: aplica blocos sequencialmente.
- `x = self.ln_final(x)`: norm final.
- `return self.lm_head(x)`: logits shape `(B, T, V)`.

**Função utilitária `count_parameters`**:
- `return sum(p.numel() for p in model.parameters())`: soma o número de elementos de cada parâmetro. *Não* checa `requires_grad` — para isso use `if p.requires_grad`.

**Bloco de teste**:
- `tiny_cfg = GPTConfig(vocab_size=VOCAB_SIZE, context_length=64, d_model=128, num_layers=2, num_heads=4, d_ff=512)`: config minúscula para smoke test.
- `tiny_model = TransformerLM(tiny_cfg).to(DEVICE)`: instancia e move.
- `print("params:", count_parameters(tiny_model))`: imprime contagem.
- `print(tiny_model(torch.randint(0, VOCAB_SIZE, (2, 16), device=DEVICE)).shape)`: forward com IDs aleatórios, shape `(2, 16)`. Imprime `(2, 16, VOCAB_SIZE)`.

```python
def run_transformer_lm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights, in_indices):
    cfg = GPTConfig(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    model = TransformerLM(cfg).to(in_indices.device)
    model.load_state_dict(weights, strict=True)
    model.eval()
    with torch.no_grad():
        return model(in_indices)
```

#### Explicação linha a linha

Esta célula é a função "stateless" usada pelos testes do assignment para rodar um forward com pesos fornecidos.

- `def run_transformer_lm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights, in_indices):`: assinatura paralela aos campos de `GPTConfig` + `weights` (state_dict) + `in_indices`.
- `cfg = GPTConfig(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)`: monta a config. Note a ordem posicional — bate com a definição da dataclass.
- `model = TransformerLM(cfg).to(in_indices.device)`: instancia e move para o mesmo device dos índices (importante: ainda não tem pesos certos).
- `model.load_state_dict(weights, strict=True)`: carrega *exatamente* os pesos passados. `strict=True` levanta erro se houver chaves faltando ou sobrando — bom para detectar typos em `state_dict`.
- `model.eval()`: desativa dropout e BN; aqui não temos nenhum, mas é hábito correto na inferência.
- `with torch.no_grad(): return model(in_indices)`: bloco sem autograd. Não vamos calcular gradientes, então pular o tracking economiza memória e tempo.

### Linha a linha do forward

- `idx` entra como IDs inteiros `(batch, seq)`.
- `token_embeddings(idx)` vira tensor contínuo `(batch, seq, d_model)`.
- `positions` explicita a posição de cada token; isso permite também usar posições deslocadas quando houver KV-cache.
- cada bloco aplica atenção causal e MLP gated com residuais.
- `ln_final` estabiliza a escala antes da cabeça.
- `lm_head` transforma cada vetor contextual em logits `(batch, seq, vocab_size)`.

O modelo não contém softmax. Logits são a interface correta para `F.cross_entropy`, porque ela aplica `log_softmax` internamente de forma estável.

> **Sanidade da contagem de parâmetros**: para um decoder-only com SwiGLU, a fórmula aproximada é
> `params ≈ V × d + L × (4 × d² + 3 × d × d_ff)` (embedding + por-bloco com 4 matrizes attention + 3 matrizes SwiGLU).
> Para Llama-style com `d_ff = 8/3 × d` e GQA fica diferente. Para esse notebook (vocab_size=4096, d_model=384, num_layers=6, num_heads=6, d_ff=1536), espere ~7-12M parâmetros. Verifique sempre depois de instanciar — discrepância entre o esperado e o real geralmente revela bug em `d_ff` ou esquecimento de uma camada.

---

## 14. Otimizador: AdamW, schedule e clipping

AdamW mantém duas estatísticas por parâmetro:

- `m`: média exponencial dos gradientes;
- `v`: média exponencial dos quadrados dos gradientes.

A correção de bias compensa o fato de essas médias começarem em zero. O `W` em AdamW significa weight decay desacoplado: em vez de misturar regularização L2 ao gradiente adaptativo, decaímos o peso separadamente.

O schedule típico para LMs pequenos:

- warmup linear no início, porque os momentos de Adam ainda estão mal estimados;
- decaimento cosseno até um LR mínimo;
- clipping global para impedir explosões raras de gradiente.

> **Nota de 2026 — Muon e a primeira mudança real desde Adam**: AdamW continua sendo o padrão para o assignment (e o "default seguro" para qualquer projeto novo), mas em maio/2026 a paisagem está em transição genuína pela primeira vez desde 2014.
>
> **Muon** (Jordan et al., final de 2024 → frontier-scale em 2025-26) ortogonaliza a matriz de atualização via Newton-Schulz antes do passo de descida. Trata pesos como matrizes com estrutura geométrica em vez de coleções escalares. Vantagens documentadas:
> - **~2× eficiência computacional** vs AdamW em treino compute-optimal;
> - **~33% menos memória** (mantém menos estado);
> - usado em produção: **Kimi K2 (1T params)** com MuonClip, **Moonlight 16B**, **GLM4.5**.
> - Quando funciona, beneficia principalmente as matrizes value-output da atenção e os blocos FFN — exatamente as memórias associativas do Transformer (Geva et al. 2020).
>
> Limitações: ortogonalização não é coordinate-wise, então requer all-gather extra em modelo paralelizado (overhead vs Adam puro). Há trabalhos de quantização (8-bit Muon) e variantes (REG, MuonBP) para mitigar.
>
> **Sophia, Lion, Adafactor, 8-bit Adam, schedule-free AdamW** — variantes para nichos específicos. Para o assignment, AdamW é o que os testes esperam; depois, vale conhecer Muon como o "next thing".

> **Nota de 2026 — schedules**: o cosine schedule abaixo continua sendo o canônico do CS336, mas a literatura empírica empurrou para **WSD (Warmup–Stable–Decay)** desde 2024. A ideia é simples: 3 fases — warmup linear, plateau constante, decay (linear-to-zero ou cosseno curto). Vantagens: (i) não precisa pré-fixar o orçamento de tokens; (ii) checkpoint da fase estável serve para continual pretraining; (iii) ablações em LLMs (Hu et al. 2024, Bergsma et al. 2025) mostram **economia de até 60% em compute** vs cosine "decay-to-10%". Para frontier-scale, WSD virou padrão. Cosine continua válido para o assignment.

```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                if wd != 0:
                    p.mul_(1 - lr * wd)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-step_size)
        return loss


def get_lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)


def clip_grad_norm(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return
    total_norm = torch.sqrt(sum(torch.sum(p.grad.detach() ** 2) for p in params))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for p in params:
            p.grad.mul_(scale)
```

#### Explicação linha a linha

Esta célula implementa AdamW, o cosine schedule e o gradient clipping — o trio canônico de otimização de LMs.

**Classe `AdamW`** (subclasse de `torch.optim.Optimizer`):
- `def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):`: hiperparâmetros padrão. Note que `betas=(0.9, 0.999)` é o default do paper original — em LMs preferimos `(0.9, 0.95)`.
- `defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)`: dicionário com os hiperparâmetros que vão para *todos* os param groups.
- `super().__init__(params, defaults)`: a base `Optimizer` cuida de organizar `params` em `param_groups`, cada um com seus próprios hiperparâmetros (permite diferentes LR por camada, p.ex.).
- `@torch.no_grad()`: o decorator desativa autograd dentro de `step` — atualizações de pesos não devem ser rastreadas.
- `def step(self, closure=None):`: API padrão. `closure` é um callable opcional usado por LBFGS; aqui só rodamos se passado.
- `loss = closure() if closure is not None else None`: idiomático.
- `for group in self.param_groups:`: itera por grupo — cada grupo pode ter LR/wd diferentes.
- Atribuições `lr, beta1, beta2, eps, wd`: extrai hiperparâmetros do grupo.
- `for p in group["params"]:`: itera por parâmetro.
- `if p.grad is None: continue`: pula parâmetros sem gradiente (congelados).
- `grad = p.grad`: alias.
- `state = self.state[p]`: dicionário persistente por parâmetro (vive entre chamadas de `step`).
- `if len(state) == 0:`: primeira vez vendo este parâmetro.
- `state["step"] = 0; state["exp_avg"] = torch.zeros_like(p); state["exp_avg_sq"] = torch.zeros_like(p)`: inicializa contador, primeiro momento (média) e segundo momento (variância) com zeros do mesmo shape e device do parâmetro.
- `exp_avg = state["exp_avg"]; exp_avg_sq = state["exp_avg_sq"]`: aliases (referências, não cópias).
- `state["step"] += 1; t = state["step"]`: avança o passo global do otimizador para este parâmetro.
- `if wd != 0: p.mul_(1 - lr * wd)`: weight decay desacoplado — *primeiro* multiplica o peso por `(1 - lr·wd)` (encolhe). Note que isso é feito *antes* da atualização de gradiente — ordem crucial.
- `exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)`: atualização in-place de `m`: `m ← β1·m + (1-β1)·g`. `add_(grad, alpha=...)` é `+= alpha * grad` em uma op.
- `exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)`: atualização in-place de `v`: `v ← β2·v + (1-β2)·g²`. `addcmul_(a, b, value=c)` faz `+= c * a * b`.
- `bias_correction1 = 1 - beta1 ** t; bias_correction2 = 1 - beta2 ** t`: corrigem o viés inicial dos momentos zerados.
- `step_size = lr * math.sqrt(bias_correction2) / bias_correction1`: junta os dois numa só constante.
- `p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-step_size)`: o passo final. `addcdiv_(a, b, value=c)` faz `+= c * a/b`. Aqui: `p -= step_size · m / (sqrt(v) + eps)` — Adam puro. `exp_avg_sq.sqrt()` cria um tensor temporário; `.add_(eps)` modifica-o in-place — note que isso *não* corrompe `exp_avg_sq` em si porque `.sqrt()` já fez cópia.
- `return loss`: hook para LBFGS.

**Função `get_lr_cosine_schedule`**:
- `if it < warmup_iters: return max_learning_rate * it / warmup_iters`: fase de warmup linear de 0 até `max_lr`.
- `if it > cosine_cycle_iters: return min_learning_rate`: depois do ciclo, mantém o mínimo.
- `progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)`: fração de progresso na fase cosseno (0 a 1).
- `cosine = 0.5 * (1 + math.cos(math.pi * progress))`: meio-cosseno descendente — vai de 1 (início) a 0 (fim).
- `return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)`: interpola entre `min_lr` (cosine=0) e `max_lr` (cosine=1).

**Função `clip_grad_norm`**:
- `params = [p for p in parameters if p.grad is not None]`: filtra parâmetros com gradiente.
- `if not params: return`: nada a fazer.
- `total_norm = torch.sqrt(sum(torch.sum(p.grad.detach() ** 2) for p in params))`: norma L2 *global* — concatenação implícita de todos os gradientes. `detach()` evita criar grafo (não é necessário em `no_grad`, mas é defensivo).
- `if total_norm > max_l2_norm:`: clipa só se passou do limite.
- `scale = max_l2_norm / (total_norm + 1e-6)`: fator de escala (<1).
- `for p in params: p.grad.mul_(scale)`: aplica em todos os gradientes in-place. Note que `total_norm` é um tensor de 0 dim — o `>` funciona com escalar Python implicitamente.

> **Intuição — bias correction**: começamos com `m = v = 0`. No início, isso enviesa as estimativas para baixo. As correções `m_hat = m / (1 - β1^t)` e `v_hat = v / (1 - β2^t)` desfazem esse viés. A versão acima fundiu as duas em `step_size = lr × sqrt(1 - β2^t) / (1 - β1^t)`, multiplicado pela atualização não corrigida, o que é equivalente. Em `t = 1`, `step_size ≈ lr × sqrt(1 - β2) / (1 - β1) ≈ lr × 0.316 / 0.1 ≈ 3.16 × lr` — Adam dá um passo *maior* logo no início. É por isso que warmup linear mata as primeiras dezenas de steps; sem ele, esses passos enormes empurram o modelo para regiões ruins.

> **Pitfall — ordem das ops em weight decay desacoplado**: a versão correta é primeiro fazer `p ← p × (1 - lr × wd)` e depois aplicar a atualização do gradiente. Se você fizer no inverso (primeiro Adam, depois decay), o efeito do decay é amplificado pela escala adaptativa de Adam — exatamente o que AdamW *evitava*. A versão acima faz na ordem correta.

### Produção

Em código real, comece com `torch.optim.AdamW` e `torch.nn.utils.clip_grad_norm_`. A implementação manual acima é útil para passar pelos detalhes do assignment, mas PyTorch cobre casos de performance, foreach/fused kernels, device placement e integração com schedulers.

Para modelos grandes, a escolha do otimizador vira decisão de sistema: AdamW consome bastante memória por causa de estados `m` e `v`; Adafactor, 8-bit Adam, Muon e sharding via ZeRO/FSDP atacam partes diferentes desse custo.

> **Quanto custa AdamW em memória**: para um modelo de N parâmetros em fp32, AdamW guarda:
> - parâmetros: 4N bytes
> - gradientes: 4N bytes
> - momento `m`: 4N bytes
> - segundo momento `v`: 4N bytes
> Total: 16N bytes só para parâmetros + estado. Para um modelo de 8B, isso é 128 GB — mais que uma H100 de 80 GB. Daí ZeRO/FSDP (sharding) e 8-bit Adam (quantização do estado). Muon reduz para ~12N por descartar o segundo momento adaptativo coordinate-wise (~25% economia direta).

---

## 15. Configurações de experimento

Não tente treinar um modelo grande de primeira. O objetivo do assignment é corretude e entendimento. Use a tabela mental:

- smoke test: 1-2 camadas, 64-128 dimensões, contexto 64;
- Colab T4/L4: 2-6 camadas, 128-384 dimensões, contexto 128-512;
- H100: 6-12 camadas, 384-768 dimensões, contexto 512-1024, bf16, batch efetivo maior.

O número mais importante para custo de treino é aproximadamente `6 * parametros * tokens`. Essa heurística conta forward + backward de um decoder dense.

> **Intuição — de onde vem o `6`**: forward custa ~`2 × params × tokens` (uma matmul por matriz, cada matmul tem 2 FLOPs por par de elementos). Backward custa ~`4 × params × tokens` (gradientes dos pesos + gradientes das ativações, ambos com mesmo trabalho do forward). Total: `6 × params × tokens`. Esta é a heurística do paper Chinchilla (Hoffmann 2022). Para Mixture-of-Experts e modelos com KV-cache na geração, fórmulas diferentes — mas para decoder dense em treino, `6 × P × T` é a estimativa usada em todos os scaling laws.

```python
@dataclass
class TrainConfig:
    context_length: int = 128
    batch_size: int = 16
    grad_accum_steps: int = 4
    max_iters: int = 200
    eval_interval: int = 50
    eval_batches: int = 10
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    compile_model: bool = False


def make_model_config(vocab_size: int, profile: str = "smoke") -> GPTConfig:
    if profile == "h100_small":
        return GPTConfig(vocab_size=vocab_size, context_length=512, d_model=512, num_layers=8, num_heads=8, d_ff=2048)
    if profile == "colab_l4":
        return GPTConfig(vocab_size=vocab_size, context_length=256, d_model=384, num_layers=6, num_heads=6, d_ff=1536)
    return GPTConfig(vocab_size=vocab_size, context_length=128, d_model=128, num_layers=2, num_heads=4, d_ff=512)


model_cfg = make_model_config(VOCAB_SIZE, profile="smoke")
train_cfg = TrainConfig(context_length=model_cfg.context_length, max_iters=20 if DEVICE == "cpu" else 100)
print(model_cfg)
print(train_cfg)
```

#### Explicação linha a linha

Esta célula define o `TrainConfig` (dataclass com hiperparâmetros de treino) e um helper para escolher modelos por perfil de hardware.

**Dataclass `TrainConfig`**:
- `context_length: int = 128`: comprimento de janela.
- `batch_size: int = 16`: sequências por step.
- `grad_accum_steps: int = 4`: gradient accumulation — o batch *efetivo* é `batch_size × grad_accum_steps = 64`. Útil quando a GPU não comporta o batch desejado em um único forward.
- `max_iters: int = 200`: total de optimizer steps.
- `eval_interval: int = 50`: a cada N iters, roda avaliação.
- `eval_batches: int = 10`: quantos batches para estimar a loss de validação (média).
- `lr: float = 3e-4`: LR pico (default canônico para LMs pequenos).
- `min_lr: float = 3e-5`: 10% do pico — convenção comum em cosine.
- `warmup_iters: int = 20`: 10% do total — também canônico.
- `weight_decay: float = 0.1`: forte (LMs grandes usam ~0.1; CV pequenos usam ~0.01).
- `grad_clip: float = 1.0`: norma máxima global.
- `compile_model: bool = False`: flag para `torch.compile`.

**Função `make_model_config`**:
- `if profile == "h100_small": return GPTConfig(vocab_size=vocab_size, context_length=512, d_model=512, num_layers=8, num_heads=8, d_ff=2048)`: perfil para GPU com bastante VRAM.
- `if profile == "colab_l4": return GPTConfig(vocab_size=vocab_size, context_length=256, d_model=384, num_layers=6, num_heads=6, d_ff=1536)`: perfil intermediário.
- `return GPTConfig(vocab_size=vocab_size, context_length=128, d_model=128, num_layers=2, num_heads=4, d_ff=512)`: smoke test default — minúsculo.

**Bloco final**:
- `model_cfg = make_model_config(VOCAB_SIZE, profile="smoke")`: monta config de modelo.
- `train_cfg = TrainConfig(context_length=model_cfg.context_length, max_iters=20 if DEVICE == "cpu" else 100)`: monta config de treino. Força `context_length` a bater com o do modelo (importante!) e reduz `max_iters` em CPU para acelerar.
- `print(model_cfg); print(train_cfg)`: imprime ambos para registro.

```python
def estimate_training_flops(params: int, tokens: int) -> float:
    return 6.0 * params * tokens


def tokens_per_step(cfg: TrainConfig) -> int:
    return cfg.batch_size * cfg.context_length * cfg.grad_accum_steps


tmp_model = TransformerLM(model_cfg)
params = count_parameters(tmp_model)
planned_tokens = tokens_per_step(train_cfg) * train_cfg.max_iters
print("params:", params)
print("tokens/step:", tokens_per_step(train_cfg))
print("planned tokens:", planned_tokens)
print("rough train FLOPs:", f"{estimate_training_flops(params, planned_tokens):.2e}")
del tmp_model
```

#### Explicação linha a linha

Esta célula calcula uma estimativa rápida de FLOPs e tokens por step — cabeça pra frente para você não treinar no escuro.

- `def estimate_training_flops(params: int, tokens: int) -> float: return 6.0 * params * tokens`: heurística canônica `6 × P × T` para decoder denso. O 6 vem de 2 (forward) + 4 (backward).
- `def tokens_per_step(cfg: TrainConfig) -> int: return cfg.batch_size * cfg.context_length * cfg.grad_accum_steps`: número de tokens efetivos por *optimizer step* (incluindo accumulation).
- `tmp_model = TransformerLM(model_cfg)`: instancia um modelo só para contar parâmetros. Não vai para device — economia de VRAM no smoke.
- `params = count_parameters(tmp_model)`: total de parâmetros.
- `planned_tokens = tokens_per_step(train_cfg) * train_cfg.max_iters`: total de tokens vistos durante o treino inteiro.
- `print("params:", params)`: imprime contagem.
- `print("tokens/step:", tokens_per_step(train_cfg))`: imprime tokens por step.
- `print("planned tokens:", planned_tokens)`: tokens planejados.
- `print("rough train FLOPs:", f"{estimate_training_flops(params, planned_tokens):.2e}")`: FLOPs estimados em notação científica (`.2e` = 2 dígitos após a vírgula, expoente).
- `del tmp_model`: libera memória do modelo de cálculo. Em GPU, isso só importa se você instanciou em device — em CPU a coleta de lixo cuidaria sozinha, mas `del` é explícito.

> **Heurística Chinchilla**: na "compute-optimal frontier" original, `tokens ≈ 20 × params`. Isto é, um modelo de 100M parâmetros pede ~2B tokens para usar bem o compute. Trabalhos pós-Llama 3 (2024-25) mostraram que para *inferência-optimal* (modelos que vão ser servidos muito), vale treinar com `tokens >> 20 × params` — Llama 3 8B foi treinado com 15T tokens (~1875× params). Para o assignment você não vai chegar nem perto, mas é útil para entender por que modelos modernos parecem "small but trained forever".

---

## 16. Avaliação e loop de treino

O loop robusto precisa fazer algumas coisas sempre:

- colocar modelo em `train()` durante treino e `eval()` durante avaliação;
- usar `torch.no_grad()` na avaliação;
- acumular gradiente dividindo a loss por `grad_accum_steps`;
- aplicar schedule antes do step;
- clipar gradiente depois do backward acumulado;
- zerar gradiente com `set_to_none=True` para economizar escrita de memória;
- salvar checkpoints com modelo, otimizador, iteração e configs.

```python
@torch.no_grad()
def estimate_loss(model: nn.Module, cfg: TrainConfig) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split, data in [("train", train_ids), ("valid", valid_ids)]:
        losses = []
        for _ in range(cfg.eval_batches):
            xb, yb = get_batch(data, cfg.batch_size, cfg.context_length, DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.reshape(-1))
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, model_cfg: GPTConfig, train_cfg: TrainConfig) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    payload = torch.load(path, map_location=DEVICE)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["iteration"])
```

#### Explicação linha a linha

Esta célula define avaliação e checkpointing — duas peças que você não enxerga até precisar.

**Função `estimate_loss`**:
- `@torch.no_grad()`: decorator desativa autograd no escopo da função inteira.
- `def estimate_loss(model: nn.Module, cfg: TrainConfig) -> dict[str, float]:`: assinatura.
- `model.eval()`: muda modo (sem efeito aqui, mas hábito correto).
- `out: dict[str, float] = {}`: dicionário de saída `{split: loss}`.
- `for split, data in [("train", train_ids), ("valid", valid_ids)]:`: avalia em ambos os splits.
- `losses = []`: lista de losses para média.
- `for _ in range(cfg.eval_batches):`: amostra `cfg.eval_batches` batches independentes.
- `xb, yb = get_batch(data, cfg.batch_size, cfg.context_length, DEVICE)`: pega batch.
- `with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):`: ativa mixed precision em GPU. `enabled=False` em CPU evita o overhead do contexto.
- `logits = model(xb)`: forward.
- `loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.reshape(-1))`: achata para `(B·T, V)` e `(B·T,)` e calcula a loss média. `.view(-1, V)` e `.reshape(-1)` são equivalentes aqui (`reshape` é tolerante a non-contiguous).
- `losses.append(loss.item())`: tira do GPU para CPU como Python float (`.item()` força sincronização).
- `out[split] = float(np.mean(losses))`: média.
- `model.train()`: volta ao modo de treino (importante!).
- `return out`: dicionário `{"train": ..., "valid": ...}`.

**Função `save_checkpoint`**:
- `def save_checkpoint(path, model, optimizer, iteration, model_cfg, train_cfg) -> None:`: assinatura completa.
- `payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration, "model_cfg": asdict(model_cfg), "train_cfg": asdict(train_cfg)}`: dicionário com tudo. Note `asdict(...)` converte dataclass em dict serializável.
- `torch.save(payload, path)`: persiste em formato pickle (`.pt`). PyTorch usa pickle por padrão; é flexível mas amarrado a versões — para portabilidade, considere `safetensors`.

**Função `load_checkpoint`**:
- `payload = torch.load(path, map_location=DEVICE)`: carrega. `map_location=DEVICE` realoca tensores para o device atual mesmo se foram salvos em outro (essencial em retomar em outra máquina).
- `model.load_state_dict(payload["model"])`: restaura pesos.
- `optimizer.load_state_dict(payload["optimizer"])`: restaura estado (`m`, `v`, `step`, LR atual).
- `return int(payload["iteration"])`: devolve o número da última iteração salva — usado no loop de treino para retomar.

> **Pitfall — `set_to_none=True`**: `optimizer.zero_grad(set_to_none=True)` *desaloca* `.grad` em vez de zerar. Para AdamW puro, ambos funcionam, mas `set_to_none=True` é mais rápido (uma escrita de memória a menos por parâmetro) e é o default desde PyTorch 2.0. Excepcionalmente, alguns custom optimizers assumem `.grad` sempre presente — cheque a doc se você usar algo fora do core.

```python
def train(model_cfg: GPTConfig, train_cfg: TrainConfig, use_manual_adamw: bool = False):
    model = TransformerLM(model_cfg).to(DEVICE)
    if train_cfg.compile_model and hasattr(torch, "compile") and DEVICE == "cuda":
        model = torch.compile(model)

    opt_cls = AdamW if use_manual_adamw else torch.optim.AdamW
    optimizer = opt_cls(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=(0.9, 0.95), eps=1e-8)

    history = []
    t0 = time.time()
    for it in range(train_cfg.max_iters + 1):
        lr = get_lr_cosine_schedule(it, train_cfg.lr, train_cfg.min_lr, train_cfg.warmup_iters, train_cfg.max_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % train_cfg.eval_interval == 0 or it == train_cfg.max_iters:
            losses = estimate_loss(model, train_cfg)
            elapsed = time.time() - t0
            history.append({"iter": it, "lr": lr, **losses, "elapsed_s": elapsed})
            print(f"iter {it:04d} | train {losses['train']:.3f} | valid {losses['valid']:.3f} | lr {lr:.2e} | {elapsed:.1f}s")

        if it == train_cfg.max_iters:
            break

        optimizer.zero_grad(set_to_none=True)
        for _ in range(train_cfg.grad_accum_steps):
            xb, yb = get_batch(train_ids, train_cfg.batch_size, train_cfg.context_length, DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.reshape(-1))
                loss = loss / train_cfg.grad_accum_steps
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

    ckpt_path = WORKDIR / "pindorama_gpt_smoke.pt"
    save_checkpoint(ckpt_path, model, optimizer, train_cfg.max_iters, model_cfg, train_cfg)
    print("checkpoint:", ckpt_path)
    return model, history


# Para estudar sem gastar GPU, o default roda pouco. Aumente max_iters depois.
model, history = train(model_cfg, train_cfg)
```

#### Explicação linha a linha

Esta é a função principal de treino — combina tudo: modelo, otimizador, schedule, autocast, accumulation, clipping, eval e checkpoint.

- `def train(model_cfg: GPTConfig, train_cfg: TrainConfig, use_manual_adamw: bool = False):`: assinatura. `use_manual_adamw` permite alternar entre nossa implementação e a oficial.
- `model = TransformerLM(model_cfg).to(DEVICE)`: instancia e move.
- `if train_cfg.compile_model and hasattr(torch, "compile") and DEVICE == "cuda": model = torch.compile(model)`: compila com `torch.compile` se disponível e em GPU. `hasattr` defensivo para versões antigas de PyTorch.
- `opt_cls = AdamW if use_manual_adamw else torch.optim.AdamW`: escolhe a classe de otimizador.
- `optimizer = opt_cls(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, betas=(0.9, 0.95), eps=1e-8)`: instancia. Note `betas=(0.9, 0.95)` — convenção LM, não o default de PyTorch.
- `history = []`: lista de logs de avaliação.
- `t0 = time.time()`: cronômetro.
- `for it in range(train_cfg.max_iters + 1):`: loop principal — `max_iters + 1` para incluir a avaliação final.
- `lr = get_lr_cosine_schedule(it, train_cfg.lr, train_cfg.min_lr, train_cfg.warmup_iters, train_cfg.max_iters)`: calcula LR para esta iter.
- `for group in optimizer.param_groups: group["lr"] = lr`: aplica o LR em todos os grupos do otimizador. (Para LR diferenciado por camada você manteria múltiplos grupos.)
- `if it % train_cfg.eval_interval == 0 or it == train_cfg.max_iters:`: faz eval periódico + ao final.
- `losses = estimate_loss(model, train_cfg)`: roda eval.
- `elapsed = time.time() - t0`: tempo decorrido.
- `history.append({"iter": it, "lr": lr, **losses, "elapsed_s": elapsed})`: dict logging — `**losses` espalha as chaves "train"/"valid".
- `print(f"iter {it:04d} | train {losses['train']:.3f} | ...")`: log formatado. `:04d` zero-padding em 4 dígitos; `:.3f` 3 decimais; `:.2e` notação científica.
- `if it == train_cfg.max_iters: break`: na última iter, só avaliamos e saímos (sem fazer mais um step).
- `optimizer.zero_grad(set_to_none=True)`: zera gradientes desalocando tensores (mais rápido que escrever zeros).
- `for _ in range(train_cfg.grad_accum_steps):`: gradient accumulation — acumula gradientes de N micro-batches antes de fazer um step.
- `xb, yb = get_batch(train_ids, train_cfg.batch_size, train_cfg.context_length, DEVICE)`: micro-batch.
- `with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):`: mixed precision.
- `logits = model(xb)`: forward.
- `loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.reshape(-1))`: cross-entropy.
- `loss = loss / train_cfg.grad_accum_steps`: divide para que a média acumulada ao longo dos micro-batches seja a média correta. *Esquecer essa divisão* é um bug clássico — a loss "aparente" fica `grad_accum_steps×` maior.
- `loss.backward()`: backward acumula em `.grad`.
- `torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)`: clip global *depois* da accumulation (não antes, ou clipa o errado).
- `optimizer.step()`: passo do otimizador.
- `ckpt_path = WORKDIR / "pindorama_gpt_smoke.pt"`: caminho do checkpoint.
- `save_checkpoint(ckpt_path, model, optimizer, train_cfg.max_iters, model_cfg, train_cfg)`: persiste.
- `print("checkpoint:", ckpt_path)`: log.
- `return model, history`: devolve modelo treinado e histórico.
- `model, history = train(model_cfg, train_cfg)`: dispara o treino com as configs definidas anteriormente.

> **Nota de 2026 — `torch.compile` em maio/2026**: PyTorch 2.11 (lançado em março/2026) é a versão atual no ecossistema. `torch.compile` é estável e o default em código novo, mas tem caveats:
> - Primeiro forward depois de `compile` pode levar 1-3 minutos para JIT (TorchDynamo + TorchInductor).
> - Mudar shapes dinamicamente recompila (use `dynamic=True` ou padding fixo).
> - Combina bem com FlexAttention para máscaras custom.
> - Em CPU, ganha pouco vs eager; em GPU, geralmente 1.3-2× em modelos médios.
>
> Para o assignment, deixe `compile_model=False` no smoke test (overhead não vale a pena) e ative para runs longos em GPU.

> **Pitfall — beta2=0.95 em LMs**: o default de PyTorch é `betas=(0.9, 0.999)`. Em LMs, a comunidade convergiu para `betas=(0.9, 0.95)` (GPT-3, Llama 2/3, Chinchilla). A motivação: gradientes em LMs têm bursts ocasionais (raros tokens, batches duros) e `β2=0.999` "lembra" demais esses bursts, causando atualizações erráticas dezenas de steps depois. `β2=0.95` esquece em ~20 steps; `β2=0.999` esquece em ~1000. O notebook acima já usa `0.95` — bom.

---

## 17. Geração: temperatura, top-k e top-p

Na geração autoregressiva, repetimos:

1. cortar o contexto para o tamanho máximo;
2. rodar o modelo;
3. pegar logits da última posição;
4. ajustar por temperatura;
5. filtrar distribuição;
6. amostrar próximo token;
7. anexar e repetir.

Temperatura baixa deixa a distribuição mais concentrada. `top_k` remove tudo exceto os k tokens mais prováveis. `top_p` mantém o menor conjunto cuja massa acumulada passa de `p`. Para modelos pequenos, geração literária ainda será fraca; o objetivo aqui é validar a pilha.

> **Intuição — temperatura como inverso de "confiança"**: dividir logits por T < 1 amplifica diferenças (deixa o softmax mais peaked); T > 1 atenua. T=0 é argmax (greedy). Em modelos calibrados, T=1 amostra "como o modelo realmente pensa". Para texto criativo, T=0.7-1.0; para código, T=0.2-0.4 ou greedy. O *combinado* temperature + top_p é o que a maioria das APIs expõe (OpenAI, Anthropic) — o uso típico é fixar p=0.9 ou 0.95 e variar só temperatura.

> **Nota de 2026 — sampling além de top-k/top-p**:
> - **min-p sampling** (Nguyen 2024, mainstream em 2025): mantém tokens com prob ≥ `p × max_prob`. Adapta a "largura" ao quanto o modelo está confiante: se top-1 tem 80%, só os tokens com ≥ 0.8 × p ficam; se top-1 tem 5%, mantém muito mais. Funciona melhor que top-p em alta temperatura porque resiste a "cauda longa explosiva".
> - **DRY (Don't Repeat Yourself)** e **XTC (Exclude Top Choices)**: heurísticas anti-repetição usadas em geração criativa local (ex: oobabooga).
> - **Speculative decoding**: técnica de inferência (não sampling): um modelo pequeno propõe múltiplos tokens; o grande verifica em paralelo. Acelera 2-5× sem mudar a distribuição final. É a razão pela qual APIs frontier são tão rápidas em 2026.
>
> Para o assignment, top-k + top-p é mais que suficiente. Min-p é uma melhoria barata — vale conhecer.

```python
def top_k_top_p_filter(logits: Tensor, top_k: int | None = None, top_p: float | None = None) -> Tensor:
    logits = logits.clone()
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        cutoff = values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < cutoff, -float("inf"))
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return logits


@torch.no_grad()
def generate(model: nn.Module, tokenizer, prompt: str, max_new_tokens: int = 80, temperature: float = 0.9, top_k: int | None = 50, top_p: float | None = 0.95) -> str:
    model.eval()
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    eot = tokenizer.token_to_id(SPECIAL_TOKEN) if hasattr(tokenizer, "token_to_id") else None
    context = model_cfg.context_length
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context:]
        logits = model(idx_cond)[:, -1, :]
        logits = logits / max(temperature, 1e-6)
        logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
        if eot is not None and next_id.item() == eot:
            break
    return tokenizer.decode(idx[0].tolist())


print(generate(model, tokenizer, "No meio do caminho", max_new_tokens=60))
```

#### Explicação linha a linha

Esta célula implementa filtragem top-k/top-p e o loop autoregressivo de geração.

**Função `top_k_top_p_filter`**:
- `def top_k_top_p_filter(logits, top_k=None, top_p=None) -> Tensor:`: assinatura.
- `logits = logits.clone()`: clona para não mutar o tensor original (importante porque vamos fazer `masked_fill_` virtualmente). Sem isso, mexer in-place poderia corromper grafo se houvesse autograd.
- `if top_k is not None and top_k > 0:`: filtragem top-k.
- `values, _ = torch.topk(logits, min(top_k, logits.size(-1)))`: pega os `top_k` maiores. `min(top_k, V)` evita pedir mais elementos do que existem.
- `cutoff = values[:, -1].unsqueeze(-1)`: o k-ésimo maior por linha. `unsqueeze(-1)` adiciona dim para broadcast.
- `logits = logits.masked_fill(logits < cutoff, -float("inf"))`: zera (em log-space) tudo abaixo do cutoff.
- `if top_p is not None and 0 < top_p < 1:`: filtragem top-p (nucleus).
- `sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)`: ordena descendente. `sorted_idx` mapeia índices ordenados de volta aos originais.
- `probs = F.softmax(sorted_logits, dim=-1)`: calcula probabilidades.
- `cumulative = torch.cumsum(probs, dim=-1)`: massa acumulada (vai de ~0 até 1).
- `remove = cumulative > top_p`: máscara booleana — `True` onde o token deve ser descartado.
- `remove[:, 1:] = remove[:, :-1].clone()`: shift right para nunca remover o top-1. `.clone()` evita aliasing in-place.
- `remove[:, 0] = False`: garante que o primeiro nunca é removido.
- `sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))`: aplica máscara nos logits ordenados.
- `logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)`: dispersa os logits filtrados *de volta* à ordem original via `sorted_idx`. `scatter_` é o inverso de `gather`.
- `return logits`: tensor filtrado pronto para softmax+amostragem.

**Função `generate`**:
- `@torch.no_grad()`: sem autograd na geração.
- `def generate(model, tokenizer, prompt, max_new_tokens=80, temperature=0.9, top_k=50, top_p=0.95) -> str:`: assinatura com defaults sensatos.
- `model.eval()`: modo de inferência.
- `ids = tokenizer.encode(prompt).ids`: tokeniza o prompt.
- `idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)`: shape `(1, T_prompt)`. `dtype=torch.long` é exigido por `nn.Embedding`.
- `eot = tokenizer.token_to_id(SPECIAL_TOKEN) if hasattr(tokenizer, "token_to_id") else None`: ID de fim, se existir.
- `context = model_cfg.context_length`: tamanho máximo de contexto.
- `for _ in range(max_new_tokens):`: gera N tokens.
- `idx_cond = idx[:, -context:]`: corta a janela mais recente para não passar do contexto suportado.
- `logits = model(idx_cond)[:, -1, :]`: forward, pega só logits da última posição. Shape: `(1, V)`.
- `logits = logits / max(temperature, 1e-6)`: aplica temperatura. `max(..., 1e-6)` evita divisão por zero quando `temperature=0` (greedy).
- `logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)`: filtra.
- `probs = F.softmax(logits, dim=-1)`: probabilidades finais.
- `next_id = torch.multinomial(probs, num_samples=1)`: amostra. Devolve shape `(1, 1)`.
- `idx = torch.cat([idx, next_id], dim=1)`: anexa o token novo.
- `if eot is not None and next_id.item() == eot: break`: para se gerou fim-de-documento.
- `return tokenizer.decode(idx[0].tolist())`: decodifica a primeira (única) linha de IDs como string. `.tolist()` converte tensor em lista Python.

**Linha final**:
- `print(generate(model, tokenizer, "No meio do caminho", max_new_tokens=60))`: gera 60 tokens a partir do verso de Drummond. Como o modelo é minúsculo e mal treinado, espere algo balbuciante — mas estruturalmente válido (UTF-8 íntegro, sem crashes).

> **Pitfall — o "shift right" do top-p**: o trecho `remove[:, 1:] = remove[:, :-1].clone()` e `remove[:, 0] = False` é a manobra crítica. Sem ele, se `top_p=0.9` e o token mais provável já tem 0.95 de massa, você removeria *todos* os tokens. O shift garante que pelo menos o top-1 sempre fica (massa não-zero antes de cruzar `p`). O `.clone()` evita aliasing in-place.

> **Pitfall — KV-cache ausente**: o `generate` acima re-computa o forward inteiro a cada novo token. Para `T = 100` tokens gerados, isso é O(100²) = O(10000) atenções em vez de O(100) com KV-cache. Para o assignment está OK; em produção, sempre use KV-cache (modelos como `transformers`'s `model.generate()` fazem isso automaticamente). Implementar KV-cache vira essencial nos próximos assignments do CS336.

---

## 18. Como isso vira uma solução local do assignment

O notebook acima contém as funções conceituais. Para entregar ou testar contra o repositório `assignment1-basics-main`, você normalmente cria módulos Python e liga os adaptadores em `tests/adapters.py`.

Mapeamento direto:

- `run_linear` -> `in_features @ weights.T`
- `run_embedding` -> `weights[token_ids]`
- `run_swiglu` -> `F.silu(x @ w1.T) * (x @ w3.T) @ w2.T`
- `run_rope` -> `RotaryEmbedding(...)(...)`
- `run_scaled_dot_product_attention` -> implementação manual com máscara booleana ou `F.scaled_dot_product_attention`
- `run_multihead_self_attention*` -> projeção QKV, split heads, RoPE opcional, causal SDPA, merge heads
- `run_transformer_block` e `run_transformer_lm` -> classes acima com `load_state_dict`
- `run_get_batch`, `run_softmax`, `run_cross_entropy`, `run_gradient_clipping`, `get_adamw_cls`, `run_get_lr_cosine_schedule`, checkpoints -> funções das seções 4, 6, 14 e 16

Se você quiser transformar isso em pacote, o próximo passo limpo é mover classes e funções para `assignment1-basics-main/cs336_basics/` e deixar o notebook apenas como narrativa + chamadas.

> **Estrutura de pacote sugerida**:
> ```
> cs336_basics/
>   __init__.py
>   tokenizer.py        # SimpleByteTokenizer, BPE training
>   data.py             # encode_texts, get_batch
>   layers.py           # Linear, Embedding, RMSNorm, SwiGLU, RotaryEmbedding
>   attention.py        # MultiHeadSelfAttention
>   model.py            # TransformerBlock, TransformerLM, GPTConfig
>   optim.py            # AdamW, get_lr_cosine_schedule, clip_grad_norm
>   training.py         # estimate_loss, save/load_checkpoint, train
>   sampling.py         # top_k_top_p_filter, generate
>   utils.py            # softmax, cross_entropy, count_parameters
> tests/
>   adapters.py         # imports + run_* shims
>   ...
> ```
> Cada módulo deve ser importável independentemente. `adapters.py` pega o que cada teste pede e passa adiante. Esta separação compensa quando você for fazer assignment 2 (systems) — você reusa o pacote inteiro.

---

## 19. Multi-GPU: DDP, FSDP e o que mudaria

Para este assignment, uma GPU é suficiente. Mas é importante saber onde as abstrações mudam.

**DDP** replica o modelo inteiro em cada GPU. Cada processo recebe batches diferentes; no backward, gradientes são sincronizados com all-reduce. O batch global é `batch_per_gpu * grad_accum * num_gpus`.

**FSDP/ZeRO** shardam parâmetros, gradientes e/ou estados do otimizador. Isso é necessário quando o modelo não cabe inteiro em uma GPU. O custo é mais comunicação e mais cuidado com checkpointing.

**Tensor parallelism** divide matmuls dentro das camadas. É útil quando uma única camada é grande demais ou quando queremos usar várias GPUs dentro de um nó NVLink/NVSwitch.

**Pipeline parallelism** divide camadas entre GPUs. É mais difícil de usar bem porque cria bolhas de pipeline.

> **Nota de 2026 — paralelismo 4D e 5D**: o estado da arte para frontier-scale combina:
> - **Data parallelism (DP)**: pelo batch.
> - **Tensor parallelism (TP)**: dentro de matmul.
> - **Pipeline parallelism (PP)**: pelas camadas.
> - **Sequence parallelism (SP)** ou **Context parallelism (CP)**: pela sequência (essencial para contexto >1M).
> - **Expert parallelism (EP)**: por expert em MoE.
>
> Para Llama 3 405B treinou com DP × TP × PP. Para DeepSeek V3 (671B MoE) e Kimi K2 (1T MoE), adicionou EP. FSDP é o "ZeRO-3 no PyTorch" e cobre DP+sharding. Megatron-LM, NeMo, e DeepSpeed implementam o resto. Para o assignment você não vai além de DDP; assignment 2 do CS336 já trabalha sistemas distribuídos a sério.

```python
DDP_TEMPLATE = r'''
# torchrun --nproc_per_node=8 train.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

local_rank = setup_ddp()
model = TransformerLM(model_cfg).to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler ou shard manual do dataset.
# A loss continua igual; DDP sincroniza gradientes durante backward().
'''
print(DDP_TEMPLATE)
```

#### Explicação linha a linha

A célula é um *template em string* — não executa código, só imprime o esqueleto que você usaria em um script real.

- `DDP_TEMPLATE = r'''...'''`: raw triple-quoted string. O `r` desliga interpretação de `\n` etc., útil quando o template tem strings com escapes literais (não é o caso aqui, mas é hábito seguro).
- `# torchrun --nproc_per_node=8 train.py`: comentário com o comando que dispara o script em 8 GPUs. `torchrun` é o launcher canônico de PyTorch; ele instancia uma cópia do processo por GPU e injeta variáveis de ambiente como `LOCAL_RANK`, `WORLD_SIZE`, `RANK`.
- `import os, torch, torch.distributed as dist`: imports de DDP. `dist` traz primitivas como `init_process_group`, `all_reduce`, `barrier`.
- `from torch.nn.parallel import DistributedDataParallel as DDP`: o wrapper que sincroniza gradientes via all-reduce.
- `def setup_ddp():`: helper.
- `dist.init_process_group(backend="nccl")`: inicializa o grupo de comunicação. NCCL é o backend NVIDIA (GPU↔GPU via NVLink/IB); para CPU usaria "gloo".
- `local_rank = int(os.environ["LOCAL_RANK"])`: o índice da GPU local nesta máquina (0..7 numa máquina com 8 GPUs).
- `torch.cuda.set_device(local_rank)`: define a GPU default para este processo.
- `return local_rank`: devolve para uso adiante.
- `local_rank = setup_ddp()`: roda o setup.
- `model = TransformerLM(model_cfg).to(local_rank)`: instancia e move para a GPU correta.
- `model = DDP(model, device_ids=[local_rank])`: envolve com DDP. Agora cada `loss.backward()` vai disparar um all-reduce sobre os gradientes (rolagem assíncrona, sobreposta ao backward).
- `# Use DistributedSampler ou shard manual do dataset.`: comentário lembrando que os splits precisam ser disjuntos por rank, ou os processos vão ver os mesmos dados.
- `# A loss continua igual; DDP sincroniza gradientes durante backward().`: nota de sanidade.
- `print(DDP_TEMPLATE)`: imprime o template para inspeção visual.

---

## 20. Profiling e sanidade de performance

CS336 insiste em accounting porque intuição sem medida costuma falhar. Sempre registre:

- tokens por segundo;
- utilização de memória GPU;
- loss train/valid;
- tokens treinados;
- configuração exata do modelo;
- versão de PyTorch/CUDA;
- dtype e `torch.compile`.

Em GPU, o primeiro passo pode ser mais lento por compilação/caches. Meça depois do warmup. Em Colab, a máquina pode variar; salve logs junto do checkpoint.

```python
def benchmark_forward(model: nn.Module, cfg: TrainConfig, steps: int = 10) -> None:
    model.eval()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(steps):
            xb, _ = get_batch(train_ids, cfg.batch_size, cfg.context_length, DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):
                _ = model(xb)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    toks = steps * cfg.batch_size * cfg.context_length
    print("forward tokens/s:", round(toks / elapsed, 1))
    if DEVICE == "cuda":
        print("max memory GB:", round(torch.cuda.max_memory_allocated() / 1e9, 3))


benchmark_forward(model, train_cfg, steps=3)
```

#### Explicação linha a linha

Função pequena para medir tokens/segundo no forward — primeiro passo de qualquer profiling.

- `def benchmark_forward(model: nn.Module, cfg: TrainConfig, steps: int = 10) -> None:`: assinatura. `steps=10` é o default para média estável.
- `model.eval()`: modo inferência.
- `if DEVICE == "cuda": torch.cuda.synchronize()`: força a GPU a terminar tudo o que está em fila *antes* de começarmos a cronometrar — essencial porque chamadas CUDA são assíncronas.
- `start = time.time()`: timestamp de início.
- `with torch.no_grad():`: sem autograd (queremos forward puro, sem custo de tracking).
- `for _ in range(steps):`: loop de medição.
- `xb, _ = get_batch(train_ids, cfg.batch_size, cfg.context_length, DEVICE)`: pega batch (descartando `y` — não precisamos).
- `with torch.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):`: mixed precision em GPU.
- `_ = model(xb)`: forward, descarta resultado.
- `if DEVICE == "cuda": torch.cuda.synchronize()`: sincroniza de novo *depois* do loop — agora `time.time()` mede o trabalho real da GPU.
- `elapsed = time.time() - start`: tempo total.
- `toks = steps * cfg.batch_size * cfg.context_length`: tokens processados.
- `print("forward tokens/s:", round(toks / elapsed, 1))`: throughput. Em uma A100 espere ~50-200k tokens/s para um modelo dessa escala; em CPU bem menos.
- `if DEVICE == "cuda": print("max memory GB:", round(torch.cuda.max_memory_allocated() / 1e9, 3))`: pico de VRAM. `max_memory_allocated` retorna *bytes*; dividimos por `1e9` para GB.
- `benchmark_forward(model, train_cfg, steps=3)`: chama com 3 steps (rápido, demonstrativo).

> **Por que `torch.cuda.synchronize()`**: chamadas CUDA são assíncronas — o `for` loop em Python termina antes da GPU realmente terminar de processar. Sem sincronizar, `time.time()` mede só o tempo de enfileiramento (microsegundos). Sincronizar força o CPU a esperar a GPU. Faça isso *antes* e *depois* da janela medida; nunca dentro do loop, ou você mata o pipelining.

> **Ferramentas de profiling 2026**:
> - **PyTorch Profiler** (`torch.profiler.profile`): traça CPU/GPU events, integra com TensorBoard.
> - **NVIDIA Nsight Systems**: profile system-wide, visualiza CUDA streams, NVLink, PCIe.
> - **NCU (Nsight Compute)**: profile kernel-by-kernel, mostra ocupância, memory bandwidth, throughput.
> - **W&B/MLflow**: tracking de métricas de treino. Você usa W&B (visto que tem MCP do W&B aqui), bom hábito mantido.

---

## 21. Próximos experimentos bons

Experimentos que ensinam mais do que simplesmente "aumentar tudo":

- Treine tokenizers 4k, 8k, 16k, 32k e compare tokens/byte, loss e velocidade.
- Mantenha FLOPs aproximadamente constantes e varie profundidade vs largura.
- Compare `d_ff = 4*d_model` contra múltiplos arredondados para GPU.
- Compare `torch.optim.AdamW` contra a implementação manual apenas em smoke tests.
- Rode com e sem `torch.compile` e meça tokens/s.
- Aumente contexto e observe o custo quadrático de atenção.
- Faça overfit em um subconjunto minúsculo para validar que o modelo aprende antes de gastar treino maior.

Regra prática: se um modelo pequeno não consegue overfit em poucos documentos, há bug de implementação, LR, batching, máscara causal ou tokenização.

> **Experimentos extras para 2026**:
> - **Substitua AdamW por Muon** (use `pip install muon-optimizer`). Compare convergência por step e memória de pico para o mesmo modelo Pindorama. Esta é uma versão controlada do experimento que Kimi K2 fez em escala 1T.
> - **WSD vs cosine**: implemente WSD (warmup → constante → linear decay-to-zero nas últimas 20% iters). Compare loss final com cosine para o mesmo orçamento de tokens.
> - **GQA**: implemente n_kv_heads = n_q_heads / 4. Confirme que a loss é semelhante com KV-cache ~4× menor. (Na inferência mesmo assim você precisa adaptar.)
> - **Min-p sampling**: substitua top-p por min-p e compare qualidade subjetiva da geração em PT-BR.
> - **RoPE θ scaling**: treine com θ=10k em contexto 256, depois rode em contexto 512 com θ=10k vs θ=40k (NTK-aware) e meça perplexidade. Veja a degradação por extrapolação.
>
> Estes ablations, em escala Pindorama-pequena, replicam exatamente o tipo de pergunta que CS336 e papers de scaling fazem em escala maior — e são reproduzíveis em uma única GPU.

---

## 22. Checklist final

Antes de confiar no notebook:

- JSON do `.ipynb` abre sem erro.
- Todas as células de código compilam.
- O fallback roda sem internet.
- O tokenizer de produção preserva acentos e faz roundtrip.
- `x` e `y` têm offset de um token.
- logits têm shape `(batch, seq, vocab_size)`.
- a loss inicial fica perto de `log(vocab_size)` para pesos aleatórios.
- a loss cai em um smoke train.
- checkpoint salva e carrega.
- geração não aplica softmax antes de `F.cross_entropy`.

Esse é o nível mínimo de higiene para um material de estudo que também serve como base de engenharia.

---

## Apêndice A — Mapa de leitura recomendada (maio/2026)

Como você está em modo "atravessar a literatura para o seu próprio trabalho de pesquisa", aqui está a versão curta da bibliografia atualizada:

**Arquitetura de transformers modernos**:
- Vaswani et al. 2017 — *Attention is All You Need* (canônico).
- Touvron et al. 2023, 2024 — Llama / Llama 2 / Llama 3 papers (RMSNorm + SwiGLU + RoPE + GQA).
- DeepSeek-AI 2024-25 — *DeepSeek V2/V3* (MLA + MoE + MTP).

**Tokenização**:
- Sennrich et al. 2016 — *BPE* original.
- Liu et al. 2025 (COLM) — *SuperBPE*.
- Pagnoni et al. 2024 — *Byte Latent Transformer (BLT)*.

**Posicionamento**:
- Su et al. 2021 — *RoFormer / RoPE*.
- Peng et al. 2023 (rev. 2026) — *YaRN*.
- Ding et al. 2024 — *LongRoPE*.

**Atenção eficiente**:
- Dao et al. 2022, 2023 — *FlashAttention 1, 2*.
- Shah et al. 2024 — *FlashAttention 3*.
- *FlexAttention* (PyTorch blog, 2024).

**Otimização**:
- Loshchilov & Hutter 2019 — *AdamW*.
- Jordan et al. 2024 — *Muon*; Bernstein 2025 — *Spectral descent*.
- Hu et al. 2024 — *Warmup-Stable-Decay*.
- Bergsma et al. 2025 — *Decay-to-zero schedules*.

**Scaling laws e treino**:
- Kaplan et al. 2020 — scaling laws originais.
- Hoffmann et al. 2022 — *Chinchilla*.
- Liu et al. 2025 — Muon scaling laws.

**CS336 oficial**:
- `cs336.stanford.edu` (Spring 2026, ativo).
- `github.com/stanford-cs336/lectures` (atualizado em 2026).
- Lecture playlist no YouTube (Spring 2025 — substantivamente igual ao Spring 2026 para o A1).

---

## Apêndice B — Diagnóstico rápido de bugs comuns

| Sintoma | Causa provável | Como diagnosticar |
|---|---|---|
| Loss inicial >> log(V) | Inicialização errada; logits explodindo | Print `logits.std()` no primeiro forward |
| Loss inicial = log(V) mas não cai | LR baixo demais ou warmup longo demais; máscara causal vazando | Reduza warmup; teste sem máscara para confirmar que ao menos overfita |
| Loss cai e depois explode | LR alto, ou bf16 com gradiente saturando | Reduza LR pela metade; cheque `torch.isnan(loss)` |
| Loss baixa mas geração ruim | Tokenizer não está fazendo roundtrip; sampling com bug | Encode→decode num texto qualquer; tente greedy (T=0) |
| `RuntimeError: view size...` | Esqueceu `.contiguous()` depois de `transpose` | Adicione `.contiguous()` antes de `.view()` |
| `device-side assert triggered` | Token ID fora do vocab; índice negativo em embedding | `assert (idx >= 0).all() and (idx < V).all()` |
| Memória explode em context maior | Atenção O(T²); KV-cache acumulando | Use `F.scaled_dot_product_attention(..., is_causal=True)`; em geração, fixe limite |
| Modelo não overfita 32 sequências | Bug em máscara, embedding ou cross_entropy | Treine com 1 sequência e veja se loss vai a zero |

---

*Versão markdown enriquecida deste notebook produzida em 9 de maio de 2026. Inserções pedagógicas e fact-checks adicionados com base na literatura atualizada até essa data — ver Apêndice A para fontes principais.*
