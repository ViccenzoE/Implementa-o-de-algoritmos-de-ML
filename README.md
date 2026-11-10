# Comparação de algoritmos de classificação

Projeto de portfólio que compara classificadores no **Wisconsin Diagnostic Breast
Cancer Dataset**. O objetivo é prever se uma amostra é benigna ou maligna e
mostrar tanto o uso responsável do scikit-learn quanto a implementação dos
fundamentos matemáticos.

## Modelos

- Regressão Logística, KNN, Decision Tree, Random Forest e SVM (scikit-learn).
- Regressão Logística implementada manualmente com NumPy, usando sigmoid,
  entropia cruzada e descida do gradiente em lote.

O dataset vem de `sklearn.datasets` e possui 569 amostras e 30 atributos. O
pipeline usa divisão estratificada 80/20, padronização nos modelos sensíveis à
escala e `random_state=42` para resultados reproduzíveis. A classe positiva é
**malignant** (maligna), portanto recall e F1 devem ser interpretados com foco
na detecção de casos malignos.

## Como executar

```bash
/bin/python3 -m pip install -r requirements.txt
/bin/python3 -m src.train
```

Os artefatos são gravados em `results/`:

- `metrics.csv`: accuracy, precision, recall e F1-score.
- `metrics_comparison.png`: comparação visual das métricas.
- `confusion_matrices.png`: uma matriz de confusão por modelo.

Para validar a implementação:

```bash
/bin/python3 -m pytest -q
```

## Estrutura

```text
src/
  logistic_regression_numpy.py  # implementação manual
  train.py                      # dados, treino, avaliação e gráficos
tests/                          # testes unitários e de integração
```

Este projeto é educacional e não deve ser usado para diagnóstico médico real.
