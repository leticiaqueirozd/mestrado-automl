import time

import matplotlib.pyplot as plt
import pandas as pd
from river import compose, linear_model, naive_bayes, preprocessing

from jitsdp.stream import load_cabral_arff, stream_from_cabral_arff
from jitsdp.evaluation import run_prequential
from jitsdp.orb import ORB
from jitsdp.base import RiverLearner

PROJETO = "fabric"

# as configuracoes que vao "competir": mesma escola (ORB), alunos diferentes
CONFIGS = {
    "ORB + Hoeffding Tree": lambda: ORB(seed=1),
    "ORB + Naive Bayes": lambda: ORB(
        base_learner=RiverLearner(naive_bayes.GaussianNB()), seed=1),
    "ORB + Regressao Logistica": lambda: ORB(
        base_learner=RiverLearner(compose.Pipeline(
            preprocessing.StandardScaler(), linear_model.LogisticRegression())),
        seed=1),
}

df = load_cabral_arff(f"data/{PROJETO}.arff")

curvas = []
for nome, criar_modelo in CONFIGS.items():
    inicio = time.time()
    res = run_prequential(criar_modelo(), stream_from_cabral_arff(df))
    res["config"] = nome
    res["commit"] = range(len(res))      # 0, 1, 2, ... (ordem de chegada)
    curvas.append(res)
    print(f"{nome:28s} | G-mean medio {res['gmean'].mean():.3f} | {time.time() - inicio:.0f}s")

# junta tudo numa tabela so e salva
tabela = pd.concat(curvas)
tabela.to_csv(f"results/curvas_{PROJETO}.csv", index=False)

# grafico: G-mean de cada configuracao ao longo do tempo
for nome, parte in tabela.groupby("config"):
    plt.plot(parte["commit"], parte["gmean"], label=nome)
plt.xlabel("Commits (em ordem de chegada)")
plt.ylabel("G-mean")
plt.title(f"Qual configuracao ganha ao longo do tempo? ({PROJETO})")
plt.legend()
plt.savefig(f"results/curvas_{PROJETO}.png", dpi=150)
print("Grafico salvo em results/")