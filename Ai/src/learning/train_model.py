# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ast
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AutoTokenizer # ✅ AutoTokenizer 사용
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 설정 (KLUE 모델로 변경!)
# ==========================================
MODEL_NAME = "klue/bert-base" # ✅ 한국어 표준 모델
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 2e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"🚀 학습 장치: {DEVICE} (GPU가 없으면 CPU로 돌아가서 느릴 수 있어요!)")

# ==========================================
# 2. 데이터셋 클래스
# ==========================================
class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        text = str(row['cleaned_content'])
        try:
            labels = ast.literal_eval(row['label'])
        except:
            labels = [0] * 30 # 에러 방지용 기본값

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
# 3. 모델 정의
# ==========================================
class KoBERTClass(nn.Module):
    def __init__(self, num_labels):
        super(KoBERTClass, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.classifier(output.pooler_output)

# ==========================================
# 4. 실행 (메인)
# ==========================================
def run_training():
    try:
        df = pd.read_csv('final_dataset_for_ai.csv')
        print(f"✅ 데이터 로드 완료: {len(df)}건")
    except:
        print("❌ 'final_dataset_for_ai.csv' 파일이 없습니다.")
        return

    # 태그 개수 자동 확인
    sample_label = ast.literal_eval(df.iloc[0]['label'])
    num_labels = len(sample_label)
    print(f"🎯 예측할 태그 개수: {num_labels}개")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # ✅ 토크나이저 로드 (AutoTokenizer 사용)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = ReviewDataset(train_df, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = KoBERTClass(num_labels)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    print("\n🔥 학습을 시작합니다! (KLUE-BERT)")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, data in enumerate(train_loader):
            ids = data['ids'].to(DEVICE)
            mask = data['mask'].to(DEVICE)
            token_type_ids = data['token_type_ids'].to(DEVICE)
            targets = data['targets'].to(DEVICE)

            outputs = model(ids, mask, token_type_ids)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if i % 100 == 0: # 로그 너무 많아서 100번에 한번만 출력
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {i} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} 완료! 평균 Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "course_mate_model.pt")
    print("\n🎉 학습 완료! 모델 저장됨.")

if __name__ == "__main__":
    run_training()