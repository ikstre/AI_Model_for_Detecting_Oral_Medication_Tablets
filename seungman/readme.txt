[작업 요약 및 파일 설명]
1. 주요 작업 내용 (전체 기간)

알약 데이터셋 구축: 추가 데이터 정제 및 7,700여 장 이미지의 YOLO 포맷 변환

데이터 분석: 이상치 제거(BBOX오류 데이터) 및 클래스별 분포 분석(EDA)

인덱스 동기화: 글로벌 JSON 기준 56종 약물 ID 일치화 (0~55)

버그 수정: 제출용 CSV 생성 시 카테고리 ID 오프셋(+1) 오류 해결

성능 개선: 시각 효과 테스트


2. 파일별 설명

[데이터 및 전처리]

pill_preprocess_v3_robust.py: 팀장님 작업물

seungman_make_dataset_local.ipynb: 팀장님 작업물

EDA_Seungman_CLEANED_Q1Q3.ipynb: 이상치 제거 및 정제 데이터 분석

global_category_index.json: 56종 알약 클래스 ID 매핑 기준 파일

eda_seungman_add1.ipynb: AI허브 추가 데이터셋 제작 코드(json은 txt로 변환)

[모델 및 학습]

pill_detection_pipeline_v23_fixed_pv1.ipynb: 시각 효과 넣어서 학습 및 인덱스 동기화 & 버그 수정된 학습 파이프라인

best(pv1).pt: 시각 효과 학습한 가중치 파일(0.9855)

best(0.985).pt: 찬영님이 만든 가중치 파일(0.9854)

[테스트 및 결과]

seungman_visualtest.ipynb: 추론 결과 시각화 테스트 (출력값 삭제됨)

eda_seungman_graph.ipynb: 데이터 통계 시각화 결과

[기타]

1.5_seungman.ipynb: 성능 0.015를 올리기 위해 틀린 데이터 분석한 코드 

test.ipynb: 오류 확인 위해 작성한 코드