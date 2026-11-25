import numpy as np
from sklearn.base import clone as sk_clone

from jitsdp.stream import KAMEI_FEATURES


# ---------------------------------------------------------------
# ETAPA 9: adaptadores (fazem qualquer modelo falar a mesma lingua)
# ---------------------------------------------------------------

# Adaptador 1: qualquer modelo online do river
class RiverLearner:
    def __init__(self, model):
        self.model = model

    def _dict(self, x):
        return dict(zip(KAMEI_FEATURES, map(float, x)))

    def learn_one(self, x, y, w=1.0):
        try:
            self.model.learn_one(self._dict(x), int(y), w=w)
        except TypeError:
            # alguns modelos nao aceitam peso (ex.: GaussianNB):
            # entao aprendem o mesmo exemplo w vezes
            for _ in range(int(round(w))):
                self.model.learn_one(self._dict(x), int(y))

    def predict_proba_one(self, x):
        p = self.model.predict_proba_one(self._dict(x))
        return np.array([p.get(0, 0.0), p.get(1, 0.0)])

    def clone(self):
        return RiverLearner(self.model.clone())


# Adaptador 2: qualquer modelo do scikit-learn
#   guarda os exemplos e re-treina o modelo a cada `refit_every` exemplos
class SklearnBatch:
    def __init__(self, estimator, refit_every=500, max_samples=5000):
        self.estimator = estimator
        self.refit_every = refit_every            # re-treina a cada 500 exemplos
        self.max_samples = max_samples            # guarda no maximo os 5000 mais recentes
        self.X, self.y, self.w = [], [], []       # os exemplos guardados
        self.desde_ultimo = 0                     # quantos chegaram desde o ultimo treino
        self.modelo = None                        # o modelo treinado (nenhum ainda)

    def learn_one(self, x, y, w=1.0):
        # guarda o exemplo
        self.X.append(np.asarray(x, float))
        self.y.append(int(y))
        self.w.append(float(w))

        # se passou do limite, esquece o mais antigo
        if len(self.y) > self.max_samples:
            self.X.pop(0); self.y.pop(0); self.w.pop(0)

        # hora de re-treinar?
        self.desde_ultimo += 1
        if self.modelo is None or self.desde_ultimo >= self.refit_every:
            if len(set(self.y)) == 2:             # precisa ter visto as duas classes
                self.modelo = sk_clone(self.estimator)
                self.modelo.fit(np.vstack(self.X), self.y, sample_weight=self.w)
                self.desde_ultimo = 0

    def predict_proba_one(self, x):
        if self.modelo is None:                   # ainda nao treinou
            return np.zeros(2)
        p = self.modelo.predict_proba(np.asarray(x, float).reshape(1, -1))[0]
        saida = np.zeros(2)
        for classe, prob in zip(self.modelo.classes_, p):
            saida[int(classe)] = prob
        return saida

    def clone(self):
        return SklearnBatch(sk_clone(self.estimator), self.refit_every, self.max_samples)