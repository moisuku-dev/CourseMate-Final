# -*- coding: utf-8 -*-
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import json
import os
import ast
from datetime import datetime
import math

# ==========================================
# 1. 설정
# ==========================================
MODEL_NAME = "klue/bert-base"
MAX_LEN = 128
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 파일 경로
DATA_FILE = r"Ai\data\final_dataset_for_ai.csv"
MODEL_FILE = r"Ai\models\course_mate_model.pt" # 학습 완료된 모델 1개만 있으면 됨
TAGS_FILE = r"Ai\src\preprocessing\tags.json"
LOG_FILE = "model_stability_evaluation.csv"   # 결과 저장 파일명

# ==========================================
# 2. 클래스 정의
# ==========================================
if not os.path.exists(TAGS_FILE):
    print(f"오류: {TAGS_FILE} 파일이 없습니다.")
    exit()

with open(TAGS_FILE, "r", encoding="utf-8") as f:
    FINAL_TAGS = json.load(f)

class KoBERTClass(nn.Module):
    def __init__(self, num_labels):
        super(KoBERTClass, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.classifier(output.pooler_output)

class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        text = str(row['cleaned_content'])
        try: labels = ast.literal_eval(row['label'])
        except: labels = [0] * len(FINAL_TAGS)

        inputs = self.tokenizer.encode_plus(
            text, None, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', return_token_type_ids=True, truncation=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'targets': torch.tensor(labels, dtype=torch.float)
        }

# ==========================================
# 3. 평가 실행 (구간별 기록)
# ==========================================
def evaluate_by_progress():
    print(f"데이터 로딩 중...")
    df = pd.read_csv(DATA_FILE)
    
    if 'label' not in df.columns:
        from create_dataset import create_label
        df['label'] = df['tokenized_words'].apply(create_label)

    # 평가용 데이터 분리
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"전체 평가 데이터: {len(test_df)}개")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = ReviewDataset(test_df, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False) # 순서대로 평가

    # 모델 로드
    model = KoBERTClass(len(FINAL_TAGS))
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # CSV 초기화
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=["Progress(%)", "Model_Name", "Macro_F1", "Micro_F1", "Exact_Accuracy", "Timestamp"]).to_csv(LOG_FILE, index=False, encoding='utf-8-sig')

    # 구간 설정 (전체 배치의 25%, 50%, 75%, 100% 지점 계산)
    total_batches = len(dataloader)
    checkpoints = {
        int(total_batches * 0.25): 25,
        int(total_batches * 0.50): 50,
        int(total_batches * 0.75): 75,
        total_batches - 1: 100  # 마지막 배치
    }
    
    # 만약 배치가 너무 적어서 키가 겹치면 강제로 100%만 실행될 수 있음 -> 방어 코드
    if len(dataloader) < 4:
        checkpoints = {total_batches - 1: 100}

    print("평가 시작 (구간별 기록)...")
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader)):
            ids = data['ids'].to(DEVICE)
            mask = data['mask'].to(DEVICE)
            token_type_ids = data['token_type_ids'].to(DEVICE)
            targets = data['targets'].cpu().numpy()

            outputs = model(ids, mask, token_type_ids)
            preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5).astype(int)

            all_targets.extend(targets)
            all_preds.extend(preds)

            # ★ 현재 배치가 체크포인트(25, 50, 75, 100%)인지 확인
            if batch_idx in checkpoints:
                percent = checkpoints[batch_idx]
                
                # 지금까지 쌓인 데이터로 점수 계산
                macro_f1 = f1_score(all_targets, all_preds, average='macro')
                micro_f1 = f1_score(all_targets, all_preds, average='micro')
                accuracy = accuracy_score(all_targets, all_preds)
                
                print(f"[진행률 {percent}%] 누적 데이터 평가 -> Macro F1: {macro_f1:.4f}")

                # CSV 저장
                log_data = {
                    "Progress(%)": percent,
                    "Model_Name": os.path.basename(MODEL_FILE),
                    "Macro_F1": round(macro_f1, 4),
                    "Micro_F1": round(micro_f1, 4),
                    "Exact_Accuracy": round(accuracy, 4),
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                pd.DataFrame([log_data]).to_csv(LOG_FILE, index=False, mode='a', header=False, encoding='utf-8-sig')

    print(f"\n 모든 평가 완료! 결과 파일: {LOG_FILE}")

if __name__ == "__main__":
    evaluate_by_progress()