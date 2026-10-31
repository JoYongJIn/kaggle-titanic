"""
titanic : 표 형태 데이터 이진 분류 신경망 (PyTorch MLP)

출력 규약
  - `--run-dir` 폴더에 metrics.jsonl(에폭별 지표), summary.json, submission.csv, model.pt 저장
  - 하이퍼파라미터는 모두 CLI 인자 (웹 UI와 Kaggle 노트북 양쪽에서 같은 방식으로 전달)
  - 데이터 경로는 `--data-dir` (Kaggle 노트북에서는 /kaggle/input/<competition>)

직접 실행:
  .venv/bin/python train.py --run-dir runs/manual --lr 0.001
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "Survived"
ID_COLUMN = "PassengerId"


# ---------------------------------------------------------------------------
# 인자
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=os.environ.get("RUN_DIR", "runs/manual"))
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    p.add_argument("--train-file", default="train.csv")
    p.add_argument("--test-file", default="test.csv")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", default="64,32", help="은닉층 크기, 쉼표 구분")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args, _unknown = p.parse_known_args()  # 노트북 커널이 넘기는 낯선 인자는 무시
    return args


# ---------------------------------------------------------------------------
# 전처리
# ---------------------------------------------------------------------------
# >>> FEATURES
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """타이타닉 전용 파생 변수."""
    df = df.copy()

    # 이름에서 호칭(Mr, Mrs, Miss, Master) 추출. 나이/성별/신분 정보를 함께 담고 있음
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = df["Title"].where(df["Title"].isin({"Mr", "Mrs", "Miss", "Master"}), "Rare")

    # 가족 크기 = 형제/배우자 + 부모/자식 + 본인
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # 객실 번호가 있는지 여부 (있으면 대체로 상위 등급)
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # 나이는 호칭별 중앙값으로 채움 (Master는 아이, Mr는 성인 등)
    df["Age"] = df["Age"].fillna(df.groupby("Title")["Age"].transform("median"))
    df["Embarked"] = df["Embarked"].fillna("S")

    # 고유값이 많은 문자열 열은 쓰지 않음
    return df.drop(columns=["Name", "Ticket", "Cabin"])
# <<< FEATURES


MAX_ONEHOT_UNIQUE = 30  # 고유값이 이보다 많은 문자열 열은 버림 (이름, 티켓번호 등)


def build_matrix(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train/test 를 함께 처리해 같은 열 구조의 숫자 행렬로 만든다."""
    tr = add_features(train)
    te = add_features(test)
    both = pd.concat([tr, te], axis=0, ignore_index=True)
    both = both.drop(columns=[c for c in (TARGET, ID_COLUMN) if c in both.columns])

    numeric, categorical = [], []
    for c in both.columns:
        s = both[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 12:
            numeric.append(c)
        elif s.nunique(dropna=True) <= MAX_ONEHOT_UNIQUE:
            categorical.append(c)
        # 그 외(고유값이 많은 문자열)는 사용하지 않음

    X = both[numeric].astype(float)
    X = X.fillna(X.iloc[: len(tr)].median())  # 학습 데이터 중앙값으로 결측 채움
    if categorical:
        cat = both[categorical].astype("string").fillna("NA")
        X = pd.concat([X, pd.get_dummies(cat, dtype=float)], axis=1)
    return X.iloc[: len(tr)].reset_index(drop=True), X.iloc[len(tr):].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """입력 -> [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Linear(1)"""

    def __init__(self, in_dim: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    hidden = tuple(int(h) for h in str(args.hidden).split(",") if h.strip())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    metrics_file = open(run_dir / "metrics.jsonl", "w")

    def log_epoch(row: dict) -> None:
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()

    # 데이터
    train_df = pd.read_csv(data_dir / args.train_file)
    test_df = pd.read_csv(data_dir / args.test_file)
    print(f"train {train_df.shape}, test {test_df.shape}")

    X_train_df, X_test_df = build_matrix(train_df, test_df)
    feature_columns = list(X_train_df.columns)
    y_all = train_df[TARGET].to_numpy(dtype=np.float32)
    print(f"features: {len(feature_columns)}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_df.to_numpy(dtype=np.float32), y_all,
        test_size=args.val_size, stratify=y_all, random_state=args.seed,
    )
    scaler = StandardScaler().fit(X_tr)
    to_t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)
    X_tr_t, y_tr_t = to_t(scaler.transform(X_tr)), to_t(y_tr).unsqueeze(1)
    X_val_t, y_val_t = to_t(scaler.transform(X_val)), to_t(y_val).unsqueeze(1)
    X_test_t = to_t(scaler.transform(X_test_df.to_numpy(dtype=np.float32)))

    # 모델
    model = MLP(X_tr_t.shape[1], hidden, args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=max(args.patience // 3, 1))
    print(model)

    @torch.no_grad()
    def evaluate(X, y):
        model.eval()
        logits = model(X)
        return criterion(logits, y).item(), ((logits > 0).float() == y).float().mean().item()

    best_val_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    n = X_tr_t.shape[0]
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i: i + args.batch_size]
            if len(idx) < 2:  # BatchNorm은 1개짜리 배치를 못 다룸
                continue
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            optimizer.step()

        tr_loss, tr_acc = evaluate(X_tr_t, y_tr_t)
        val_loss, val_acc = evaluate(X_val_t, y_val_t)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        log_epoch({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
            "train_acc": tr_acc, "val_acc": val_acc, "lr": lr_now, "time": round(time.time() - t0, 2),
        })

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f} | lr {lr_now:.2e}")
        if stale >= args.patience:
            print(f"early stop at epoch {epoch}")
            break

    metrics_file.close()
    model.load_state_dict(best_state)
    val_loss, val_acc = evaluate(X_val_t, y_val_t)
    print(f"\nbest epoch {best_epoch} -> val loss {val_loss:.4f}, val acc {val_acc:.3f}")

    # 예측 및 저장
    model.eval()
    with torch.no_grad():
        pred = (model(X_test_t) > 0).int().squeeze(1).cpu().numpy()
    submission = pd.DataFrame({ID_COLUMN: test_df[ID_COLUMN], TARGET: pred})
    submission.to_csv(run_dir / "submission.csv", index=False)
    torch.save({"state_dict": best_state, "feature_columns": feature_columns, "hidden": hidden,
                "dropout": args.dropout}, run_dir / "model.pt")

    (run_dir / "summary.json").write_text(json.dumps({
        "best_epoch": best_epoch, "epochs_run": epoch, "val_loss": val_loss, "val_acc": val_acc,
        "n_features": len(feature_columns), "test_positive_rate": float(pred.mean()),
        "seconds": round(time.time() - t0, 1),
    }, indent=2))
    print(f"saved {run_dir / 'submission.csv'} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
