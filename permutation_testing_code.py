# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 11:29:13 2025

"""


"""

High throughput PCA + SVM classification workflow with:
•	StratifiedGroupKFold cross-validation o preserve group structure across folds
•   Grouping variable (ID) 
•   Uses 'class' as the label column.
•	PCA component sweep (1 → ncomp) - PCA is fit only on training data and then applied to the test set to avoid data leakage.
•	Hyperparameter optimisation via GridSearchCV - applied separately for each number of components to find the best SVM hyperparameters.
    
    - Records:
        * Train CV macro F1 (best per inner CV)
        * Test macro/micro/weighted F1
        * Accuracy, balanced accuracy, precision, recall
        * Best SVM hyperparameters (C, kernel, gamma)
    
•   Saves per-fold metrics CSV for each fold/component/seed.
•   Saves per-sample predictions + class probabilities CSV.
•   Uses multiple seeds (default 20) for robustness.
•   Verbose progress printing for monitoring.
•	A restricted permutation test. 

"""
# Load packages
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


# Run Controls / Settings
DETERMINISTIC = False
CV_SPLITS = 5
N_SEEDS = 20
START_SEED = 1
CONFUSION_MATRIX_MODE = "aggregate_final_only"
PROGRESS_EVERY = 5

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", OUTPUT_DIR / "checkpoints"))
INPUT_CSV = Path("./file.csv") # Data file - change as appropriate

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# Load & preprocess
data = pd.read_csv(INPUT_CSV)
# Label encoding of the class variable
label_enc = LabelEncoder()
data["label"] = label_enc.fit_transform(data["class"])
class_names = list(label_enc.classes_)
n_classes = len(class_names)

#create spectral matrix - drop metadata cols
drop_cols = {
    "ID", "class", "label", 
    "confounder"
}
drop_cols = [c for c in drop_cols if c in data.columns]

X_df = data.drop(columns=drop_cols)
X_df = X_df.select_dtypes(include=[np.number])
X_df = X_df.fillna(X_df.median(numeric_only=True))

var0 = X_df.columns[X_df.nunique() <= 1].tolist()
if var0:
    X_df = X_df.drop(columns=var0)

X = X_df
y = data["label"]
groups = data["ID"]


# Permutation utilities
def confounder_shuffling(df, conf_name):
    df_new = df.copy()
    df_new[conf_name] = np.random.permutation(df_new[conf_name].values)
    return df_new


def restricted_response_shuffling(df, label_name, conf_name):
    df_new = df.copy()
    for level in df[conf_name].unique():
        idx = df[df[conf_name] == level].index
        df_new.loc[idx, label_name] = np.random.permutation(df.loc[idx, label_name].values)
    return df_new


def run_pca_svm_auc(X_train, y_train, X_test, y_test, n_components, paramspace, seed):
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_train)

    X_tr_red = pca.transform(X_train)
    X_tst_red = pca.transform(X_test)

    svm_mod = SVC(probability=True, random_state=seed)
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    grid = GridSearchCV(
        svm_mod,
        param_grid=paramspace,
        cv=cv_inner,
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid.fit(X_tr_red, y_train.ravel())

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    y_proba = best_model.predict_proba(X_tst_red)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    return {
        "auc": auc,
        "best_model": best_model,
        "best_params": best_params,
        "pca": pca,
    }


def conf_perm_test_auc_pca_svm(
    df, feat_cols, label_col, conf_col,
    train_idx, test_idx, nperm, n_components,
    paramspace, seed
):
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    X_train = df_train[feat_cols].values
    y_train = df_train[label_col].values
    X_test = df_test[feat_cols].values
    y_test = df_test[label_col].values

    obs = run_pca_svm_auc(
        X_train, y_train, X_test, y_test,
        n_components=n_components,
        paramspace=paramspace,
        seed=seed,
    )
    auc_obs = obs["auc"]

    perm_null = np.zeros(nperm)
    #independnet permutation seed offset
    NULL_OFFSET = 100000

    for i in range(nperm):
        # independent permutation seed
        null_seed = NULL_OFFSET + seed * 1000 + i
    
        np.random.seed(null_seed)
        random.seed(null_seed) 
        
        df_train_p = confounder_shuffling(df_train, conf_col)
        df_test_p = confounder_shuffling(df_test, conf_col)

        df_train_s = restricted_response_shuffling(df_train_p, label_col, conf_col)
        df_test_s = restricted_response_shuffling(df_test_p, label_col, conf_col)

        X_train_s = df_train_s[feat_cols].values
        y_train_s = df_train_s[label_col].values
        X_test_s = df_test_s[feat_cols].values
        y_test_s = df_test_s[label_col].values

        res_s = run_pca_svm_auc(
            X_train_s, y_train_s, X_test_s, y_test_s,
            n_components=n_components,
            paramspace=paramspace,
            seed=null_seed,
        )
        perm_null[i] = res_s["auc"]

    return {
        "auc_obs": auc_obs,
        "perm_null": perm_null,
        "npermr": len(df_test),
        "obs_result": obs,
    }


# Hyperparameters
paramspace = {
    "C": [0.1, 0.5, 0.7, 0.9, 1],
    "kernel": ["rbf", "poly", "linear", "sigmoid"],
    "gamma": ["scale", "auto"],
}
ncomp = 50

PER_FOLD_METRICS_PATH = OUTPUT_DIR / "foldwise_metrics.csv"
PER_SAMPLE_PRED_PATH = OUTPUT_DIR / "per_sample_predictions.csv"

pd.DataFrame(
    columns=[
        "seed", "fold", "epoch", "pca_component",
        "svm_C", "svm_kernel", "svm_gamma",
        "f1_macro_cv_train_best", "f1_macro_test",
        "f1_micro_test", "f1_weighted_test",
        "accuracy", "balanced_accuracy",
        "precision_macro", "recall_macro",
        "skipped_reason",
    ]
).to_csv(PER_FOLD_METRICS_PATH, index=False)

pd.DataFrame(
    columns=[
        "seed", "fold", "row_index", "ID",
        "true_label", "true_label_enc",
        "pred_label", "pred_label_enc",
        "pca_component", "svm_C", "svm_kernel", "svm_gamma",
    ]
).to_csv(PER_SAMPLE_PRED_PATH, index=False)


# Performance tracking
combined_cm_sum = None
done = 0
total_settings = N_SEEDS * CV_SPLITS * ncomp
print(f"[INFO] Estimated total evaluations: {total_settings}")


# Loop over seeds
for seed in range(START_SEED, START_SEED + N_SEEDS):
    random.seed(seed)
    np.random.seed(seed)

    sgkf = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        groups_test = groups.iloc[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        pca = PCA(n_components=ncomp)
        pca.fit(X_train)

        X_train_red = pca.transform(X_train)
        X_test_red = pca.transform(X_test)

        for j in range(1, ncomp + 1):
            X_tr_red = X_train_red[:, :j]
            X_tst_red = X_test_red[:, :j]

            svm_mod = SVC(probability=True)
            cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

            grid = GridSearchCV(
                svm_mod,
                param_grid=paramspace,
                cv=cv_inner,
                scoring="f1_macro",
                n_jobs=-1,
            )
            grid.fit(X_tr_red, y_train.ravel())

            best_model = grid.best_estimator_
            best_params = grid.best_params_

            y_pred = best_model.predict(X_tst_red)

            f1_train_best = grid.best_score_
            f1_macro = f1_score(y_test, y_pred, average="macro")
            f1_micro = f1_score(y_test, y_pred, average="micro")
            f1_weight = f1_score(y_test, y_pred, average="weighted")
            acc = accuracy_score(y_test, y_pred)
            bacc = balanced_accuracy_score(y_test, y_pred)
            prec_mac = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec_mac = recall_score(y_test, y_pred, average="macro", zero_division=0)

            fold_row = {
                "seed": seed,
                "fold": fold_idx,
                "epoch": 1,
                "pca_component": j,
                "svm_C": best_params.get("C"),
                "svm_kernel": best_params.get("kernel"),
                "svm_gamma": best_params.get("gamma"),
                "f1_macro_cv_train_best": f1_train_best,
                "f1_macro_test": f1_macro,
                "f1_micro_test": f1_micro,
                "f1_weighted_test": f1_weight,
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "precision_macro": prec_mac,
                "recall_macro": rec_mac,
                "skipped_reason": "",
            }

            pd.DataFrame([fold_row]).to_csv(
                PER_FOLD_METRICS_PATH, mode="a", header=False, index=False
            )

            CONF_NAME = "confounder"

            df_fold = pd.concat(
                [X_df.reset_index(drop=True),
                 data[["label", CONF_NAME]].reset_index(drop=True)],
                axis=1,
            )

            perm_result = conf_perm_test_auc_pca_svm(
                df=df_fold,
                feat_cols=X_df.columns.tolist(),
                label_col="label",
                conf_col=CONF_NAME,
                train_idx=train_idx,
                test_idx=test_idx,
                nperm=len(test_idx),
                n_components=j,
                paramspace=paramspace,
                seed=seed,
            )

            p_value = (
                np.sum(perm_result["perm_null"] >= perm_result["auc_obs"]) + 1
            ) / (len(perm_result["perm_null"]) + 1)

            print(
                f"[Permutation Test] Seed={seed}, Fold={fold_idx}, PCA={j}, "
                f"AUC_obs={perm_result['auc_obs']:.3f}, p={p_value:.4f}"
            )

            pd.DataFrame({
                "seed": seed,
                "fold": fold_idx,
                "pca_component": j,
                "perm_null": perm_result["perm_null"],
            }).to_csv(
                OUTPUT_DIR / "perm_null_full.csv",
                mode="a",
                header=not (OUTPUT_DIR / "perm_null_full.csv").exists(),
                index=False,
            )

            pd.DataFrame({
                "seed": [seed],
                "fold": [fold_idx],
                "pca_component": [j],
                "auc_obs": [perm_result["auc_obs"]],
                "perm_mean": [np.mean(perm_result["perm_null"])],
                "perm_sd": [np.std(perm_result["perm_null"])],
                "p_value": [p_value],
            }).to_csv(
                OUTPUT_DIR / "perm_test_results.csv",
                mode="a",
                header=not (OUTPUT_DIR / "perm_test_results.csv").exists(),
                index=False,
            )

            y_proba = best_model.predict_proba(X_tst_red)
            labels_test_str = label_enc.inverse_transform(y_test)

            rows = []
            for i_samp, idx in enumerate(test_idx):
                r = {
                    "seed": seed,
                    "fold": fold_idx,
                    "row_index": int(idx),
                    "ID": int(groups_test.iloc[i_samp]),
                    "true_label": labels_test_str[i_samp],
                    "true_label_enc": int(y_test.iloc[i_samp]),
                    "pred_label": label_enc.inverse_transform([y_pred[i_samp]])[0],
                    "pred_label_enc": int(y_pred[i_samp]),
                    "pca_component": j,
                    "svm_C": best_params.get("C"),
                    "svm_kernel": best_params.get("kernel"),
                    "svm_gamma": best_params.get("gamma"),
                }
                for k, cls in enumerate(class_names):
                    r[f"proba_{cls}"] = float(y_proba[i_samp, k])
                rows.append(r)

            pd.DataFrame(rows).to_csv(
                PER_SAMPLE_PRED_PATH, mode="a", header=False, index=False
            )

           

            done += 1
            if done % PROGRESS_EVERY == 0:
                print(
                    f"[PROGRESS] {done}/{total_settings} | "
                    f"Seed {seed}, Fold {fold_idx}, PC={j} | F1_macro={f1_macro:.3f}"
                )

