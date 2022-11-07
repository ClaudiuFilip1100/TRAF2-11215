import matplotlib.pyplot as plt
import pandas as pd

hist = pd.read_csv('../../data/model_history.csv')
print(hist.head())


plt.figure(figsize=(12, 12))
plt.plot(hist["accuracy"], label="accuracy")
# plt.plot(hist["val_accuracy"], label="validation acc.")
plt.legend()
plt.savefig("../../utils/Model Accuracy v2")

plt.figure(figsize=(12, 12))
plt.plot(hist["auc"], label="AUC score")
# plt.plot(hist["val_auc"], label="validation AUC score")
plt.legend()
plt.savefig("../../utils/Model AUC Score v2")

plt.figure(figsize=(12, 12))
plt.plot(hist["precision"], label="Precision score")
# plt.plot(hist["val_precision"], label="validation Precision score")
plt.legend()
plt.savefig("../../utils/Model Precision v2")

plt.figure(figsize=(12, 12))
plt.plot(hist["recall"], label="Recall score")
# plt.plot(hist["val_recall"], label="validation Recall score")
plt.legend()
plt.savefig("../../utils/Model Recall v2")