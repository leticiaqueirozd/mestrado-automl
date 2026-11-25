import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# ETAPA 6: o "corretor" e a prova
# ---------------------------------------------------------------

# o corretor: guarda a nota de cada classe, esquecendo devagar o passado
class FadingRecall:
    def __init__(self, alpha=0.99):
        self.alpha = alpha
        self.acertos = np.zeros(2)    # [acertos nos limpos, acertos nos defeituosos]
        self.total = np.zeros(2)      # [quantos limpos vistos, quantos defeituosos vistos]

    def update(self, y, pred):
        # o passado perde um pouco de peso (x alpha) e soma a questao nova
        self.acertos[y] = self.alpha * self.acertos[y] + (1.0 if pred == y else 0.0)
        self.total[y] = self.alpha * self.total[y] + 1.0

    def recalls(self):
        # R0 e R1 = acertos / total (se ainda nao viu nenhum, a nota e 0)
        return np.divide(self.acertos, self.total, out=np.zeros(2), where=self.total > 0)


# aplica a "prova" inteira: percorre a fila de eventos da Etapa 5
def run_prequential(model, events, alpha=0.99):
    corretor = FadingRecall(alpha)
    linhas = []
    for e in events:
        if e.kind == "test":
            pred = model.predict_one(e.x, time=e.time)     # o modelo chuta
            corretor.update(e.y, pred)                     # o corretor confere
            r0, r1 = corretor.recalls()
            linhas.append((e.time, e.y, pred, r0, r1, np.sqrt(r0 * r1)))
        else:
            # o rotulo chegou: o modelo aprende
            t = e.time if e.y == 1 else e.commit_time
            model.learn_one(e.x, e.y, time=t)
    return pd.DataFrame(linhas, columns=["time", "y", "pred", "r0", "r1", "gmean"])


# a nota final: a media das notas ao longo do tempo
def summarize(res):
    return {
        "gmean": round(float(res["gmean"].mean()), 3),
        "r0": round(float(res["r0"].mean()), 3),
        "r1": round(float(res["r1"].mean()), 3),
    }