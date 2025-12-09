# -*- coding: utf-8 -*-
import torch
from torch import nn
from transformers import BertModel, AutoTokenizer # ✅ 변경됨
import numpy as np
import json

# ==========================================
# 1. 설정
# ==========================================
MODEL_NAME = "klue/bert-base" # ✅ 변경됨
MAX_LEN = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    with open("tags.json", "r", encoding="utf-8") as f:
        FINAL_TAGS = json.load(f)
    print(f"✅ 태그 리스트 로드 완료! (총 {len(FINAL_TAGS)}개)")
except:
    print("❌ tags.json 파일이 없습니다.")
    exit()

class KoBERTClass(nn.Module):
    def __init__(self, num_labels):
        super(KoBERTClass, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.classifier(output.pooler_output)

def predict(text):
    print(f"\n🔄 분석 중... '{text}'")
    model = KoBERTClass(len(FINAL_TAGS))
    try:
        model.load_state_dict(torch.load("course_mate_model.pt", map_location=DEVICE))
    except:
        print("❌ 모델 파일 없음")
        return
    model.to(DEVICE)
    model.eval()

    # ✅ 토크나이저 변경
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    inputs = tokenizer.encode_plus(
        text, None, add_special_tokens=True, max_length=MAX_LEN,
        padding='max_length', return_token_type_ids=True, truncation=True
    )

    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(DEVICE)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(DEVICE)
    token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids)
    
    probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    print("-" * 35)
    for i, prob in enumerate(probs):
        if prob > 0.5:
            print(f"✅ {FINAL_TAGS[i]} ({prob*100:.1f}%)")
    print("-" * 35)

# ==========================================
# 4. 테스트 문장 넣기 (총 35개)
# ==========================================
if __name__ == "__main__":
    # 🚨 주의: 아래 리뷰를 돌리기 전에, 38,000건 학습을 먼저 완료해야 합니다!
    reviews = [
        "튀소는 진짜 맛있는데 가격도 착해서 선물하기 좋아요.", 
        "평일인데도 줄이 너무 길어서 한참 기다렸다가 들어갔네요 ㅠㅠ", 
        "대전 오면 무조건 들러야 하는 필수 코스입니다!",
        "음식 양이 엄청 많고 맛도 있어서 여러 명이 와도 배 터지게 먹겠네. 주차도 편리해서 만족!", 
        "직원들이 너무 친절하게 안내해 줘서 기분 좋았고, 매장이 넓어서 답답함이 없었어요.", 
        "여기는 일몰 뷰가 정말 멋있고 사진 찍기 좋은 곳이 많아서 인생샷 건졌어요.", 
        "아이들 교육용 전시가 잘 되어 있어서 가족끼리 방문하기 딱 좋아요.", 
        "혼자 와서 창가 자리에서 노트북 켜고 카공하기 좋은 분위기였어요.", 
        "건물이 오래됐는데 레트로 감성이 살아있어서 독특했고, 커피 맛은 좋았어요.", 
        "강아지랑 같이 들어갈 수 있는 펫프렌들리 카페라 정말 좋았어요.", 
        "비가 와서 실내 데이트 코스를 찾았는데, 내부 시설이 깨끗하고 좋았어요.", 
        "여긴 진짜 힙하고 트렌디해서 20대들이 많이 올 것 같네요.", 
        "대중교통으로 오기 너무 편한 역 근처라 접근성이 좋았어요.", 
        "조용하고 한적한 곳이라 복잡한 거 싫어하는 사람에게 힐링 코스입니다.", 
        "고급스러운 분위기에 직원도 친절해서 기념일 데이트 코스로 완벽합니다.", 
        "음식은 맛있지만 가격이 조금 비싼 편이라 자주 오긴 부담스러워요. 직원은 매우 친절했습니다.", 
        "주말에 갔는데도 의외로 손님이 별로 없고 조용해서 책 읽기 딱 좋았어요.", 
        "이곳은 입장료가 따로 없어서 부담없이 와서 시간 보내기 좋았어요.", 
        "메뉴판에는 없지만 시그니처 디저트를 꼭 드셔보세요. 양도 엄청 넉넉해서 만족했어요.", 
        "매장이 아주 깨끗하고 위생관리가 철저해서 어린아이와 함께 오기 좋겠어요.", 
        "오래된 건물인데 인테리어가 빈티지 무드로 바뀌어서 사진 찍는 재미가 있었어요.", 
        "친한 친구랑 둘이 와서 수다 떨기 좋았고, 분위기가 젊은 층에게 트렌디하게 먹힐 것 같아요.", 
        "비가 와서 야외 활동은 못했지만 실내에서 즐길 거리가 많아서 다행이었어요.", 
        "역에서 걸어서 3분 거리라 대중교통으로 오기 완벽했고, 근처에 벤치도 많아 잠시 힐링하기 좋았어요.", 
        "이국적인 느낌의 조형물이 많아서 마치 해외여행 온 기분이었어요.", 
        "주차 공간이 여유롭고 넓어서 초보 운전자도 편하게 주차할 수 있었어요.", 
        "커플끼리 와서 로맨틱한 분위기를 즐기기에 최고의 장소입니다. 강력 추천!", 
        "단체 모임하기 좋게 테이블이 붙어있었고, 공간이 넓어서 다른 손님 방해 없이 대화 가능했어요.", 
        "반려견 동반이 가능해서 댕댕이랑 같이 뷰 좋은 곳에서 커피 마시고 왔습니다.", 
        "박물관 체험이 아이 교육에 좋았고, 기념품샵도 있어서 가족 여행으로 만족했습니다.", 
        "화장실까지 깔끔하고 냄새가 안 나서 기분 좋게 이용했어요. 관리가 잘 된 클린 공간!", 
        "어르신이나 몸이 불편한 사람도 엘리베이터가 있어 접근성이 우수했어요.", 
        "사장님이 메뉴 설명도 친절하게 잘 해주시고 서비스도 좋았습니다.", 
        "밤에 조명이 켜지니 낮과는 완전히 다른 분위기라 나이트 뷰 명소로 손색이 없네요.", 
        "와이파이가 빵빵 터지고 콘센트도 많아서 재택 근무나 카공하기 딱 좋아요." 
    ]
    
    for review in reviews:
        predict(review)