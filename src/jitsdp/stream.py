import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY = 86400  # quantos segundos tem um dia

# os nomes das 14 metricas, na ordem em que aparecem no arquivo
KAMEI_FEATURES = ["fix", "ns", "nd", "nf", "entrophy", "la", "ld", "lt",
                  "ndev", "age", "nuc", "exp", "rexp", "sexp"]


# ---------------------------------------------------------------
# ETAPA 4: ler o arquivo .arff e transformar em tabela
# ---------------------------------------------------------------
def load_cabral_arff(path):
    # os nomes das 17 colunas: 14 metricas + bug? + data + tipo
    colunas = KAMEI_FEATURES + ["contains_bug", "timestamp", "commit_type"]
    linhas = []

    # 1) abre o arquivo e le linha por linha
    with open(path) as arquivo:
        for linha in arquivo:
            linha = linha.strip()   # tira espacos e o "enter" do fim

            # 2) pula o que nao e dado: linha vazia, cabecalho (@) e comentario (%)
            if not linha or linha.startswith("@") or linha.startswith("%"):
                continue

            # 3) quebra a linha nas virgulas -> lista com 17 pedacos
            partes = [p.strip() for p in linha.split(",")]

            # 4) se nao tiver 17 pedacos, a linha esta quebrada: pula
            if len(partes) != 17 or partes[16] == "":
                continue

            linhas.append(partes)

    # 5) monta a tabela
    df = pd.DataFrame(linhas, columns=colunas)

    # 6) tudo chegou como texto; converte para numero
    df["fix"] = (df["fix"] == "True").astype(float)              # True/False -> 1/0
    df["contains_bug"] = (df["contains_bug"] == "True").astype(int)
    for c in KAMEI_FEATURES[1:]:
        df[c] = df[c].astype(float)
    df["timestamp"] = df["timestamp"].astype(float).astype(np.int64)
    df["commit_type"] = df["commit_type"].astype(int)
    return df


# ---------------------------------------------------------------
# ETAPA 5: o framework de waiting time
# ---------------------------------------------------------------
@dataclass
class Event:
    kind: str          # "test" ou "train"
    x: np.ndarray      # as 14 metricas do commit
    y: int             # rotulo: 0 = limpo, 1 = defeituoso
    time: int          # quando o evento acontece
    commit_time: int   # quando o commit foi feito
    idx: int           # numero da linha na tabela


# Parte B: versao que reproduz o artigo (usa os tipos 0 a 3 dos .arff)
def stream_from_cabral_arff(df):
    X = df[KAMEI_FEATURES].to_numpy(dtype=float)  # as metricas, em forma de matriz
    ts = df["timestamp"].to_numpy()
    tipos = df["commit_type"].to_numpy()

    fila = []   # commits esperando os 90 dias: (data do commit, linha)
    pos = 0     # qual e o proximo da fila a sair

    for i in range(len(df)):
        agora = int(ts[i])

        # TIPO 3: chegou a reclamacao -> aprende "defeituoso" agora
        if tipos[i] == 3:
            yield Event("train", X[i], 1, agora, agora, i)
            continue

        # libera da fila quem ja esperou 90 dias -> aprende "limpo"
        while pos < len(fila) and (agora - fila[pos][0]) // DAY >= 90:
            data_commit, j = fila[pos]
            pos += 1
            yield Event("train", X[j], 0, agora, data_commit, j)

        # TIPOS 0, 1 e 2: commit novo -> o modelo tenta adivinhar
        rotulo_real = 0 if tipos[i] == 0 else 1
        yield Event("test", X[i], rotulo_real, agora, agora, i)

        # tipos 0 e 1 entram na fila: daqui a 90 dias viram "limpo"
        if tipos[i] in (0, 1):
            fila.append((agora, i))


# Parte D: versao generica (qualquer dataset, qualquer W)
def build_waiting_time_stream(X, commit_ts, y, found_ts, waiting_days=90):
    X = np.asarray(X, dtype=float)
    W = waiting_days * DAY            # W em segundos
    eventos = []

    for i in range(len(y)):
        t = int(commit_ts[i])

        # 1) todo commit e testado no dia em que chega
        #    (tempo, desempate, linha, tipo, rotulo)
        eventos.append((t, 1, i, "test", int(y[i])))

        defeituoso = y[i] == 1 and not np.isnan(found_ts[i])
        achado_antes_de_W = defeituoso and (found_ts[i] - t) < W

        # 2) se o bug NAO foi achado antes de W -> vira "limpo" no dia t + W
        if not achado_antes_de_W:
            eventos.append((t + W, 0, i, "train", 0))

        # 3) se teve bug -> vira "defeituoso" no dia da descoberta
        if defeituoso:
            eventos.append((int(found_ts[i]), 0, i, "train", 1))

    # 4) entrega os eventos em ordem de tempo
    heapq.heapify(eventos)
    while eventos:
        t, _, i, kind, rotulo = heapq.heappop(eventos)
        yield Event(kind, X[i], rotulo, t, int(commit_ts[i]), i)