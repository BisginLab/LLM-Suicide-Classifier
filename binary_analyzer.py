import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

def to_binary(pred):
    if pred >= .6:
        return 1
    else:
        return 0
def to_inverse(pred):
    return 1 - pred

print("Loading Dataset...")
pre_label_df = pd.read_csv('~/Github repo/pred_scale_labels.csv', header=0)
label_df = pre_label_df.iloc[50000: len(pre_label_df)]
print(label_df.columns)
y_true = label_df['True'].tolist()
y_pred = label_df['Predicted'].tolist()
y_pred_bin = label_df['Predicted'].apply(to_binary).tolist()
print("Dataset Loaded...")
print(f"Size of pre datasset: {len(pre_label_df)}")
print(f"Size of dataset: {len(y_true)}")

#Get prediction results
print("Overall Metrics:")
print("Accuracy: ",accuracy_score(y_true, y_pred_bin))
print("Precision: ",precision_score(y_true, y_pred_bin))
print("Recall: ",recall_score(y_true, y_pred_bin))
print("F1: ",f1_score(y_true, y_pred_bin))
print()

print("Classification Report:")
print(classification_report(y_true, y_pred_bin))
print()

print("AUC and ROC")
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
print(f"Thresholds: {thresholds}")
roc_auc = roc_auc_score(y_true, y_pred)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.savefig('/home/umflint.edu/brayclou/Github repo/roc_curve2.png')
plt.show()
plt.close()
print("ROC curve saved as roc_curve2.png")