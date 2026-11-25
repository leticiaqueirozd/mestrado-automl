from collections import Counter, deque

import numpy as np
from river import tree

from jitsdp.stream import KAMEI_FEATURES
from jitsdp.base import RiverLearner


# ---------------------------------------------------------------
# ETAPA 7: um classificador sozinho (baseline)
# ---------------------------------------------------------------
class SingleLearner:
    def __init__(self):
        self.model = tree.HoeffdingTreeClassifier()   # a arvore do artigo

    # converte a lista de 14 numeros em dicionario {"fix": ..., "ns": ..., ...}
    def _dict(self, x):
        return dict(zip(KAMEI_FEATURES, map(float, x)))

    # o aluno chuta: 0 (limpo) ou 1 (defeituoso)
    def predict_one(self, x, time=None):
        p = self.model.predict_proba_one(self._dict(x))   # probabilidade de cada classe
        p0, p1 = p.get(0, 0.0), p.get(1, 0.0)
        if p0 + p1 == 0:      # ainda nao aprendeu nada: chuta "limpo"
            return 0
        return 0 if p0 > p1 else 1    # escolhe a classe mais provavel

    # o aluno aprende com um exemplo
    def learn_one(self, x, y, time=None):
        self.model.learn_one(self._dict(x), int(y))


# ---------------------------------------------------------------
# ETAPA 8: o ORB  (+ ETAPA 9: classificador base trocavel)
# ---------------------------------------------------------------
class ORB:
    def __init__(self, base_learner=None, n_models=10, theta=0.99, ws=100, th=0.4,
                 l0=10.0, l1=12.0, m=1.5, noise_n=3, seed=1):
        # ETAPA 9: se ninguem disser qual modelo usar, usa a Hoeffding Tree do artigo
        if base_learner is None:
            base_learner = RiverLearner(tree.HoeffdingTreeClassifier())

        # PARTE A: o comite = 10 copias do modelo base
        self.ensemble = [base_learner.clone() for _ in range(n_models)]
        self.rng = np.random.default_rng(seed)    # o "sorteador" (seed = sorteio repetivel)

        # PARTE B: o tamanho de cada classe [rho_0, rho_1], comecando meio a meio
        self.theta = theta
        self.class_size = np.array([0.5, 0.5])

        # PARTE C: o termostato
        self.past_preds = deque(maxlen=ws)        # guarda so as ultimas ws (100) predicoes
        self.th, self.l0, self.l1, self.m = th, l0, l1, m

        # PARTE C: os dois detalhes do codigo original
        self.noise_n = noise_n
        self.clean_seen = Counter()               # quantas vezes vimos cada commit limpo
        self.defect_pool = deque()                # fila de defeituosos esperando

    # ---------- o comite vota
    def predict_one(self, x, time=None):
        votos = np.zeros(2)                       # [votos para limpo, votos para defeituoso]
        for membro in self.ensemble:
            v = membro.predict_proba_one(x)       # ETAPA 9: o adaptador ja devolve [p0, p1]
            if v.sum() > 0:
                votos += v / v.sum()              # cada membro vale no maximo 1 voto
        if votos.sum() == 0:                      # ninguem aprendeu nada ainda
            return 0
        pred = 0 if votos[0] > votos[1] else 1    # ganha a classe com mais votos
        self.past_preds.append(pred)              # PARTE C: anota a resposta para o termostato
        return pred

    # ---------- PARTE C: o termostato
    def ma(self):
        # fracao de "defeituoso" nas ultimas predicoes (0 a 1)
        return float(np.mean(self.past_preds)) if self.past_preds else 0.0

    def obf(self):
        # resultado > 0: reforca os DEFEITUOSOS por esse fator
        # resultado < 0: reforca os LIMPOS pelo valor sem o sinal
        ma, th, m = self.ma(), self.th, self.m
        if ma < th:
            f = (m ** (abs(ma - th) * 10) - 1) / (m ** (th * 10) - 1) * self.l1 + 1
        else:
            f = -((m ** (ma * 10) - m ** (th * 10)) / (m ** 10 - m ** (th * 10)) * self.l0 + 1)
        return 1.0 if abs(f) < 1 else f

    # ---------- chegou um exemplo de treino
    def learn_one(self, x, y, time=None):
        chave = tuple(x)                          # as 14 metricas, para comparar commits
        if y == 1:
            # PARTE C (filtro de ruido): identico a 3+ limpos? provavelmente rotulo errado
            if self.clean_seen[chave] >= self.noise_n:
                return
            # PARTE C (fila): guarda o defeituoso para treinar junto com um limpo
            self.defect_pool.append(x)
        else:
            self.clean_seen[chave] += 1
            self._train(x, 0)                     # treina o limpo
            if self.defect_pool:                  # e libera um defeituoso da fila
                self._train(self.defect_pool.popleft(), 1)

    # ---------- cada membro aprende com um peso sorteado
    def _train(self, x, y):
        # PARTE B: atualiza o tamanho das classes
        self.class_size = self.theta * self.class_size + (1 - self.theta) * np.eye(2)[y]

        # PARTE B: lambda = classe maior / classe deste exemplo
        maior = int(np.argmax(self.class_size))
        lam = self.class_size[maior] / self.class_size[y]

        # PARTE C: o empurrao do termostato
        obf = self.obf()

        for membro in self.ensemble:
            k = int(self.rng.poisson(lam))        # sorteia o peso deste membro
            if y == 1 and obf > 0:                # PARTE C: reforca os defeituosos
                k = int(k * obf)
            if y == 0 and obf < -1:               # PARTE C: reforca os limpos
                k = int(k * -obf)
            if k > 0:
                membro.learn_one(x, y, float(k))  # ETAPA 9: aprende pelo adaptador