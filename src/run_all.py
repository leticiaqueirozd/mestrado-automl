import time

import pandas as pd

from jitsdp.stream import load_cabral_arff, stream_from_cabral_arff
from jitsdp.evaluation import run_prequential, summarize
from jitsdp.orb import ORB

# os 10 projetos do artigo, e o G-mean do codigo Java original dos autores
JAVA = {
    "fabric": 0.584, "jgroups": 0.565, "camel": 0.604, "tomcat": 0.600,
    "brackets": 0.630, "neutron": 0.798, "spring-integration": 0.515,
    "broadleaf": 0.601, "nova": 0.753, "npm": 0.550,
}

resultados = []
for projeto, gmean_java in JAVA.items():
    inicio = time.time()
    df = load_cabral_arff(f"data/{projeto}.arff")
    res = run_prequential(ORB(seed=1), stream_from_cabral_arff(df))
    nota = summarize(res)

    print(f"{projeto:20s} | G-mean {nota['gmean']:.3f} (Java {gmean_java:.3f}) "
          f"| R0 {nota['r0']:.3f} | R1 {nota['r1']:.3f} | {time.time() - inicio:.0f}s")

    resultados.append({"projeto": projeto, "gmean_java": gmean_java, **nota})

# junta tudo numa tabela, salva em CSV (abre no Excel) e mostra a media
tabela = pd.DataFrame(resultados)
tabela.to_csv("results/resultados_orb.csv", index=False)

print()
print("MEDIA | G-mean Python:", round(tabela["gmean"].mean(), 3),
      "| G-mean Java:", round(tabela["gmean_java"].mean(), 3))