# **의료 분야 AI 활용과 전망: 부민병원 세미나**

## **I. 의료 AI의 여명: 새로운 가능성의 시대**

의료 분야에서 인공지능(AI)의 역할이 변혁적으로 대두되면서, 진단, 치료부터 행정 효율화 및 환자 관리 방식에 이르기까지 광범위한 영역에서 혁신을 예고하고 있습니다.1 현대 의료 시스템이 마주한 복잡성과 증가하는 요구사항 속에서, AI는 데이터 기반의 맞춤형 예측 의료를 실현할 핵심 기술로 주목받고 있습니다. 본 세미나는 부민병원 의료진 및 임직원 여러분을 대상으로 AI의 현주소와 미래 전망에 대한 전문가적 이해를 돕고자 마련되었습니다. 이 시간을 통해 의료 AI의 기본 원리부터 실제 적용 사례, 그리고 다가올 미래의 발전 방향까지 심도 있게 논의할 것입니다. AI 기술, 특히 대규모 언어 모델(LLM)과 다중 에이전트 시스템의 발전은 의료 서비스 제공 방식을 근본적으로 변화시킬 잠재력을 지니고 있으며, 이는 의료 전문가들이 AI를 이해하고 활용하는 능력이 더욱 중요해짐을 시사합니다.3 본 세미나가 이러한 변화에 대비하고, AI를 통해 의료 서비스의 질을 한 단계 높이는 데 기여할 수 있기를 기대합니다.

## **II. 지능의 해독: 대규모 언어 모델(LLM)의 근본 원리**

대규모 언어 모델(LLM)은 인간과 유사한 텍스트를 이해하고 생성할 수 있는 강력한 인공지능의 한 형태로, 복잡한 의료 및 유전체 데이터 처리에 있어 상당한 응용 가능성을 보여주고 있습니다.1 이러한 LLM의 핵심에는 **트랜스포머(Transformer) 아키텍처**와 \*\*어텐션 메커니즘(Attention Mechanism)\*\*이 자리 잡고 있습니다.

트랜스포머 아키텍처는 기존의 순차적 데이터 처리 방식(예: RNN)에서 벗어나 병렬 처리를 가능하게 함으로써, 긴 시퀀스의 의료 문서나 유전체 서열을 더 빠르고 효율적으로 분석할 수 있게 합니다.5 이 구조의 핵심인 어텐션 메커니즘은 모델이 출력값을 생성할 때 입력 시퀀스의 여러 부분 중 어떤 부분에 더 집중해야 할지 가중치를 부여하여 결정합니다.5 이는 방대한 의료 텍스트나 유전체 서열 내에서 문맥적으로 중요한 정보를 선별하는 데 결정적인 역할을 합니다.

어텐션 메커니즘에는 여러 종류가 있는데, \*\*셀프 어텐션(Self-Attention)\*\*은 모델이 특정 단어를 처리할 때 입력 문장 내의 다른 단어들과의 관계를 파악하여 문맥적 연관성을 포착하도록 합니다.5 \*\*멀티 헤드 어텐션(Multi-Head Attention)\*\*은 이러한 셀프 어텐션을 여러 "헤드"에서 병렬적으로 수행하여, 입력 정보를 다양한 관점에서 동시에 고려함으로써 더 풍부한 특징 표현을 학습할 수 있게 합니다.5

트랜스포머는 토큰을 병렬로 처리하기 때문에 순서 정보가 부족할 수 있는데, 이를 보완하기 위해 \*\*위치 인코딩(Positional Encoding)\*\*이 사용됩니다. 이는 각 토큰의 상대적 또는 절대적 위치 정보를 모델에 제공하여 시퀀스 내 순서를 이해하도록 돕습니다.6 이러한 구성 요소들은 LLM이 긴밀한 의존 관계와 복잡한 패턴을 가진 의료 데이터를 효과적으로 처리할 수 있는 기반이 됩니다. 유전체 서열 데이터가 자연어 텍스트와 구조적으로 유사하다는 점은 NLP 분야에서 성공을 거둔 이러한 LLM 기술들이 생정보학 분야, 특히 유전체 데이터 분석에 성공적으로 적용될 수 있는 중요한 이유 중 하나입니다.5

LLM이 의료 언어를 이해하고 생성하는 과정은 입력된 텍스트를 "토큰(token)"이라는 단위로 분할하고, 이를 사전에 정의된 "어휘(vocabulary)" 집합을 기반으로 처리하는 방식으로 이루어집니다.6 트랜스포머 아키텍처는 크게 인코더-디코더 구조, 인코더 전용 구조(예: BERT), 디코더 전용 구조(예: GPT)로 나눌 수 있으며, 각각의 구조는 특정 작업(예: 번역, 문맥 이해, 텍스트 생성)에 최적화되어 활용됩니다.6

**표 1: LLM 아키텍처의 핵심 구성 요소**

| 구성 요소 | 간략 설명 | 의료/유전체 데이터에 대한 주요 이점 |
| :---- | :---- | :---- |
| 트랜스포머 (Transformer) | 병렬 처리 기반의 신경망 아키텍처 | 긴 시퀀스(의료 기록, 유전체)의 빠르고 효율적인 처리 가능 5 |
| 셀프 어텐션 (Self-Attention) | 입력 시퀀스 내 단어들 간의 문맥적 관계를 파악 | 의료 텍스트 내 복잡한 의미론적 관계 이해 향상 5 |
| 멀티 헤드 어텐션 (Multi-Head Attention) | 여러 관점에서 동시에 정보에 주목하여 다양한 특징 학습 | 데이터의 다각적 분석 및 미묘한 패턴 인식 강화 5 |
| 위치 인코딩 (Positional Encoding) | 토큰의 순서 정보를 모델에 제공 | 유전체 서열 등 순서가 중요한 데이터의 정확한 해석 지원 6 |
| 인코더/디코더 모듈 | 입력 정보의 특징 추출(인코더) 및 목표 출력 생성(디코더) | 의료 질의응답, 요약, 번역 등 다양한 작업 수행 6 |

트랜스포머의 병렬 처리와 어텐션 메커니즘 같은 아키텍처 혁신은 LLM이 이전 모델들을 능가하여 의료 및 유전체 정보의 규모와 복잡성을 효과적으로 다룰 수 있게 된 직접적인 원인입니다. 의료 데이터(유전체, EHR, 의학 문헌 등)의 양이 폭발적으로 증가함에 따라, 이러한 원칙에 기반한 LLM은 의미 있는 통찰력을 추출하는 데 있어 점점 더 필수적인 도구가 될 것입니다. 또한, 유전체학의 "언어"가 인간 언어와 구조적 유사성을 공유한다는 점은 NLP에서 파생된 LLM 기술이 생정보학에서 놀라울 정도로 효과적임을 시사하며, 이는 향후 LLM이 다양한 생물학적 데이터 유형을 서로 다른 "언어"로 취급하여 근본적인 문법과 의미론을 통해 통찰력을 연결할 수 있는 미래를 암시합니다.

## **III. 의료 현장 혁신 AI: 현재 적용 사례 및 성공담**

인공지능(AI)은 의료 현장의 다양한 영역에서 이미 실질적인 변화를 이끌어내고 있으며, 진단 정확도 향상부터 치료법 개발, 환자 관리 효율화에 이르기까지 그 영향력을 확대하고 있습니다.

**A. AI를 통한 진단 기술의 발전**

1. 의료 영상 분석 (방사선학, 병리학, 피부과학):  
   AI, 특히 합성곱 신경망(CNN)과 딥러닝 기술은 CT, MRI, X-ray, 유방촬영술 이미지, 병리 슬라이드 등 다양한 의료 영상에서 패턴을 감지하고 분류하는 데 탁월한 성능을 보입니다.7 이를 통해 이상 징후의 조기 발견율을 높이고 진단 정확도와 효율성을 향상시키는 데 기여하고 있습니다.7 신경 영상 및 흉부 영상, 그리고 CT와 MRI와 같은 영상 유형은 임상적 수요가 높고 복잡성이 크기 때문에 AI 제품 개발의 주요 대상이 되고 있습니다.7  
   구체적인 사례로, **Google Health의 연구**는 주목할 만합니다.8 유방암 검진에서 AI 모델은 임상의와 비슷하거나 더 높은 정확도로 유방촬영술 이미지를 분석했으며(Nature 발표), 전이성 유방암 진단을 위한 LYNA 도구를 개발했습니다 (Archives of Pathology & Laboratory Medicine, AJSP). 폐암 분야에서도 딥러닝을 활용하여 보다 정확한 폐암 검진 가능성을 제시했으며(Nature Medicine), 피부과 영역에서는 컴퓨터 비전 AI를 통해 수백 가지 피부, 모발, 손톱 질환을 식별하는 성과를 거두었습니다 (Nature Medicine, JAMA Network Open). 특히, 피부색이 어두운 환자들의 데이터 부족 문제를 해결하기 위해 SCIN 데이터셋을 구축하기도 했습니다. 이 외에도 안구 사진을 통해 빈혈을 감지하거나(Nature Biomedical Engineering), 망막 이미지를 분석하여 심혈관 질환 위험을 평가하는(Nature Biomedical Engineering) 연구도 진행되었습니다.  
   실제 임상 현장에서는 **FDA 승인을 받은 AI 도구**들이 활용되기 시작했습니다. **Paige Prostate**는 병리학자들이 전립선암 의심 병변을 탐지하는 데 도움을 주는 AI 시스템으로, 여러 연구에서 높은 민감도(예: 96-99%)와 특이도(예: 93-98%)를 보이며 병리학자의 진단 정확도와 효율성을 개선하는 것으로 나타났습니다.9 Paige 사는 PanCancer Detect라는 AI 도구에 대해 FDA 혁신 의료기기 지정을 받기도 했습니다.12 **Viz.ai Stroke Triage**는 CT 스캔을 분석하여 대혈관 폐색(LVO)을 신속하게 감지하고 뇌졸중 치료팀에 알림을 보내는 FDA 승인 AI 솔루션입니다.13 이를 통해 진단 및 치료까지의 시간을 단축시키는 효과가 보고되었는데, 예를 들어 LVO 진단 및 치료 담당 혈관 내 외과 의사와의 첫 접촉까지의 시간이 44% 감소하고, 병원 간 이송 시간(door-in-door-out time)이 절반으로 줄어든 사례가 있습니다.15 Viz ICH Plus는 두개내 출혈량을 정량화하는 데 사용됩니다.14 이러한 AI 도구들은 주로 인간 전문가를 대체하기보다는 보조하는 역할을 수행하며, 진료의 질을 높이는 데 기여하고 있습니다.9  
2. 전자건강기록(EHR) 기반 예측 분석:  
   전자건강기록(EHR)은 AI가 질병 위험 및 예후를 예측하는 데 필요한 방대한 데이터셋을 제공합니다.17 특히 심혈관 질환(CVD) 위험 예측 분야에서 AI/ML 모델은 EHR 데이터(약물 사용, 영상 정보, 인구 통계학적 정보 등)를 분석하여 CVD 발생 위험을 예측합니다.17 딥러닝은 예측 성능을 향상시키지만, 결과 해석의 어려움이라는 과제도 안고 있습니다. 또한, 다중이환 및 만성 질환 관리에서도 AI는 EHR 데이터를 활용하여 위험 요인 예측, 건강 데이터 추출, 동반 질환 프로파일링, 의료 자원 사용 예측 등 다양한 연구에 활용되고 있습니다 (예: 만성 신장 질환, 당뇨병, 정신 질환).18 다만, 데이터의 질, 표준화, 지역적 편차 등은 여전히 해결해야 할 주요 과제로 남아있습니다.17

**B. 치료 및 수술 혁신**

1. 정밀 방사선 종양학에서의 AI 활용:  
   AI는 복잡한 방사선 치료 계획 수립 과정을 자동화하고 최적화하여 계획 수립 시간을 단축하며 정확도를 향상시킵니다.19 머신러닝 알고리즘은 방대한 데이터셋(환자 예후, 영상 정보 등)을 분석하여 개인 맞춤형 치료 계획을 위한 최적의 치료 매개변수를 식별하며, 이를 통해 치료 효과를 극대화하고 부작용을 최소화할 수 있습니다.19 상용 시스템으로는 Varian사의 Ethos Therapy와 Elekta사의 MOSAIQ Plaza 등이 있으며, 이들은 AI를 활용하여 실시간 해부학적 변화에 따라 치료 계획을 조정하는 적응 방사선 치료(ART) 기능을 제공합니다.19 \*\*방사선체학(Radiomics)\*\*은 영상 데이터를 분석하여 치료 결과를 예측하고 개인 맞춤형 치료 계획을 수립하는 데 기여합니다.20 AI 덕분에 과거 수일이 걸리던 치료 계획 수립이 수 시간 내로 단축될 수 있게 되었습니다.20  
2. AI 기반 로봇 수술:  
   AI는 로봇 수술 시스템에 통합되어 봉합, 조직 박리 등의 작업을 자동화하고, 수술 정밀도를 높이며, 실시간 피드백을 제공하는 방향으로 발전하고 있습니다.21 Intuitive Surgical사의 다빈치(da Vinci) 5 시스템에는 수술 후 피드백을 제공하는 "Case Insights" AI 도구가 탑재되어 있으며, 향후 실시간 피드백 제공을 목표로 하고 있습니다.21 Stryker사의 Mako 시스템은 "Blueprint" AI 기술을 활용하여 어깨 및 척추 수술 전 계획 수립을 지원합니다.21 이러한 AI 기반 로봇 수술은 합병증 감소, 안전성 향상, 환자 회복 기간 단축, 입원 기간 단축 등의 이점을 제공합니다.21 미래에는 자율 로봇이 외과 의사 부족 문제를 해결하고, 의사는 자문 역할을 수행하는 모습도 그려지고 있으며, 촉각 피드백 기술 개발이 중요한 연구 분야로 남아있습니다.22 로봇 수술 시장의 경쟁 심화는 기술 혁신을 더욱 가속화하고 있습니다.21

**C. 제약 개발의 혁명**

1. AI 기반 신약 개발:  
   AI/ML 기술은 전통적인 신약 개발 과정의 높은 비용, 긴 시간, 낮은 성공률 문제를 해결할 잠재력을 가지고 있습니다.23 약물 표적 식별 단계에서는 자연어 처리(NLP) 기술이 방대한 과학 문헌을 분석하고, 머신러닝이 유망한 표적을 예측합니다.24 분자 설계 및 선도 물질 발굴에서는 생성형 AI(Generative AI)가 새로운 분자 구조를 창조하고 그 특성을 예측하며, 딥러닝, 그래프 신경망(GNN), 트랜스포머 등이 활용됩니다.23 또한 AI는 실험실에서의 실험 프로토콜 최적화를 제안하기도 합니다.24 상용 플랫폼으로는 Atomwise가 CNN과 방대한 화학 라이브러리를 결합하여 신약 후보 물질을 발굴하며 Sanofi, Bayer 등과 파트너십을 맺고 있고 25, Certara.AI는 생명 과학 분야에 특화된 AI 플랫폼으로 신약 개발, 임상 시험, 규제 제출 등에 맞춤형 GPT 기술을 제공합니다.27 AI는 신약 개발 과정에서 단순 관찰자를 넘어 능동적인 협력자로 진화하고 있습니다.24  
2. 임상 시험 최적화:  
   AI는 임상 시험 과정을 간소화하고, 비용을 절감하며, 환자 예후를 개선하는 데 기여합니다.28 시험 설계 및 맞춤화 단계에서 AI는 환자 데이터를 분석하여 임상 시험 참여자 선정/제외 기준을 조정함으로써 잠재적 모집군을 두 배로 늘리고 필요한 표본 크기를 줄일 수 있습니다.28 환자 모집 및 매칭에서는 AI가 EHR을 NLP로 스캔하여 적격 참여자를 식별하고, 예측 모델을 통해 특정 시험에 가장 적합하거나 혜택을 볼 가능성이 높은 환자를 찾아냅니다.28 데이터 분석 및 모니터링에서는 실시간 데이터 분석, 이상 감지, 데이터 일관성 유지 등을 지원합니다.29 Certara.AI 및 CODEX는 큐레이션된 임상 결과 데이터베이스와 AI를 결합하여 임상 시험 분석, 메타 분석, 시험 설계 최적화를 지원합니다.27 AI는 임상 시험 계획 및 실행의 오랜 과제들을 해결하는 데 중요한 역할을 하고 있습니다.28

**D. 환자 관리 및 참여 방식의 재구상**

1. AI 챗봇 및 가상 비서:  
   AI 챗봇은 NLP와 머신러닝을 활용하여 실시간 건강 정보 제공, 증상 평가 지원, 예약 관리, 원격 환자 모니터링 등의 서비스를 제공합니다.2 이를 통해 의료 접근성을 개선하고, 대기 시간을 줄이며, 의료 소외 지역 지원, 정신 건강 관리, 만성 질환 관리 등에 기여합니다.31 하이브리드 챗봇은 AI와 인간의 입력을 결합하여 보다 개인화된 상호작용을 제공합니다.2 다만, 복잡한 진단에서는 인간 의사를 대체할 수 없으며, 데이터 프라이버시와 알고리즘 편향 문제가 우려됩니다.31  
2. 앰비언트 클리니컬 인텔리전스 (ACI):  
   ACI는 음성 인식 AI를 사용하여 의사와 환자 간의 자연스러운 대화 중에 실시간으로 진료 내용을 EHR에 자동으로 기록하는 기술입니다.32 이를 통해 의사는 환자에게 더욱 집중할 수 있게 되어 소통과 신뢰 관계를 증진시킵니다.32 또한, 문서화의 정확성과 완전성을 높이고 업무 흐름 효율성을 개선합니다.32 LLM은 ACI에서 NLP 및 보고서 생성에 핵심적인 역할을 합니다.32 상용 솔루션인 Ambience Healthcare는 실시간 진료 기록 생성, 코딩 지원, 진료 후 요약, 의뢰서 자동 생성 등 다양한 기능을 여러 전문과에 맞춰 제공합니다.33 ACI는 행정 업무로 인한 의사의 번아웃을 줄이는 것을 목표로 합니다.3  
3. 개인 맞춤형 정밀 의료를 위한 유전체학 AI:  
   AI는 유전체 데이터(오믹스, 유전적 위험 인자(GRF) 등)와 임상 데이터를 결합 분석하여 개인 맞춤형 의료 서비스를 제공합니다.34 주요 응용 분야로는 질병 스크리닝, 환자 계층화, 신약 개발, 질병 기전 이해, 치료 반응 예측(약물유전체학), 최적 용량 결정 등이 있습니다.35 AI는 다중 모드 오믹스 데이터를 통합하여 심층적인 통찰력을 제공하며 35, 정밀 의료는 예측적, 예방적, 개인 맞춤형, 참여형 의료를 지향합니다.34 사전 동의 및 유전 정보 차별과 같은 윤리적 고려 사항이 매우 중요합니다.35

**표 2: 의료 분야 AI 적용 현황 및 전망 (현재 및 부상)**

| 의료 영역 | AI 기술/도구 예시 | 주요 적용 사례 | 간략한 미래 전망/영향 |
| :---- | :---- | :---- | :---- |
| 진단 \- 영상 | CNN, 딥러닝, Paige Prostate, Viz.ai | 암 조기 발견, 뇌졸중 진단, 피부 질환 식별 8 | 진단 정확도 및 효율성 지속적 향상, 워크플로우 최적화 |
| 진단 \- EHR | 머신러닝, NLP, LLM | 심혈관 질환 위험 예측, 만성 질환 관리 지원 17 | 질병 발생 예측 및 예방적 개입 강화, 개인 맞춤형 건강 관리 지원 |
| 치료 \- 방사선 종양학 | 머신러닝, Ethos Therapy, MOSAIQ Plaza | 개인 맞춤형 방사선 치료 계획 최적화, 적응 방사선 치료 19 | 치료 효과 극대화 및 부작용 최소화, 치료 계획 시간 단축 |
| 치료 \- 로봇 수술 | AI 통합 로봇 시스템 (da Vinci, Mako) | 수술 정밀도 향상, 자동화된 수술 보조, 실시간 피드백 21 | 수술 결과 개선, 외과 의사 업무 부담 경감, 자율 수술 시스템으로의 발전 가능성 |
| 신약 개발 | 생성형 AI, CNN, GNN, Atomwise, Certara.AI | 신규 약물 표적 식별, 분자 구조 설계, 임상 시험 최적화 23 | 신약 개발 기간 단축 및 비용 절감, 성공률 향상 |
| 환자 관리 및 참여 | NLP 기반 챗봇, ACI (Ambience Healthcare), 유전체 분석 AI | 실시간 건강 상담, 진료 기록 자동화, 개인 맞춤형 정밀 의료 31 | 환자 중심 의료 강화, 의료 접근성 향상, 의사 행정 부담 감소 |

Paige Prostate, Viz.ai와 같은 AI 도구들이 FDA 승인을 받았다는 사실은 특정 진단 영역에서 AI 기술이 연구 단계를 넘어 검증된 임상 적용 단계로 성숙했음을 의미합니다. 이는 AI의 신뢰성에 대한 잠재적 회의론에 대응하는 강력한 근거가 되며, 광범위한 채택을 위한 준비가 되었음을 시사합니다. 따라서 부민병원과 같은 의료기관은 각 전문 분야와 관련된 FDA 승인 AI 솔루션을 적극적으로 모니터링하여 임상 워크플로우에 통합할 가능성을 검토해야 합니다.

AI의 역할은 초기에는 진단 보조나 계획 지원과 같은 지원 도구에서 7, ACI를 통한 문서 자동화 32나 자율 수술의 가능성 22 등 보다 자율적인 기능으로 진화하고 있습니다. 이러한 추세는 의료진의 워크플로우 재설계, 기술 개발, 그리고 환자 치료에서 AI 자율성의 윤리적 경계에 대한 선제적인 논의를 필요로 합니다.

EHR, 유전체학, 영상 아카이브와 같은 "빅데이터"와 AI 간의 시너지는 예측 분석 및 개인 맞춤형 의학 혁신의 주요 동력입니다.8 이는 AI 알고리즘 자체에 투자하는 것만큼이나 강력한 데이터 인프라, 데이터 품질 및 데이터 거버넌스에 대한 투자가 AI를 효과적으로 활용하려는 병원에 매우 중요하다는 것을 의미합니다.

AI가 신약 개발 가속화, 치료 계획 시간 단축, 행정 부담 감소 등 상당한 효율성 향상을 제공하지만 20, 특히 환자 상호작용과 복잡한 의사 결정에서는 "인간적인 손길"이 여전히 중요합니다. 챗봇이 의사를 완전히 대체할 수 없고 31, ACI가 의사가 환자 상호작용에 더 많은 시간을 할애하도록 돕는 것을 목표로 하며 32, 많은 진단 AI가 보조적인 역할을 한다는 점 9은 협력적인 미래를 시사합니다. 따라서 AI 구현은 단순히 자동화 자체를 위한 자동화가 아니라 인간의 능력을 증강하고 의사-환자 관계를 개선하는 데 초점을 맞춰야 합니다.

## **IV. 협력적 지능: 의료 분야 다중 에이전트 시스템(MAS)**

**A. 복잡한 의료 작업을 위한 다중 에이전트 시스템(MAS) 소개**

다중 에이전트 시스템(MAS)은 단일 에이전트의 능력을 넘어서는 문제를 해결하기 위해 여러 지능형 에이전트가 협력하는 시스템을 의미합니다.38 의료 분야는 본질적으로 여러 전문가와 다양한 유형의 데이터를 포함하는 협력적인 성격을 지니고 있기 때문에, MAS는 다양한 AI 에이전트를 조정함으로써 이러한 협력적 환경을 모방할 수 있어 적합합니다.39 MAS는 의료 서비스를 더 원활하고, 빠르며, 효과적으로 만들고, 실시간 모니터링, 개인 맞춤형 치료 계획, 향상된 정보 공유 등의 이점을 제공할 수 있습니다.39

**B. 자연어 상호작용을 통한 진단 및 치료 정확도 향상**

본 세미나의 주요 관심사 중 하나는 MAS가 자연어 대화를 통해 어떻게 진단 및 치료 정확도를 향상시키는가 하는 점입니다. MAS는 환자 또는 에이전트 간의 다중 턴(multi-turn) 대화에 참여하여 포괄적인 정보를 수집하고, 이를 통해 보다 정확한 진단과 적절한 전문가 식별을 유도할 수 있습니다.38 각 에이전트는 텍스트, 이미지, 검사 결과 등 다양한 출처의 정보를 처리하고 종합하여, 그 결과를 자연어 요약 형태로 조정 에이전트나 인간 임상의에게 전달할 수 있습니다.41

**C. 주요 연구 및 사례**

1. MedAgent-Pro 41:  
   MedAgent-Pro는 다중 모드 의료 진단을 위한 증거 기반 추론 에이전트 워크플로우입니다. 이 시스템은 계층적 구조를 특징으로 합니다. \*\*작업 수준(Task Level)\*\*에서는 MLLM(Multi-modal Large Language Model)이 플래너 에이전트 역할을 수행하며, 검색된 임상 기준을 통합하여 지식 기반 추론을 통해 특정 질병에 대한 신뢰할 수 있는 진단 계획을 생성합니다.42 \*\*사례 수준(Case Level)\*\*에서는 여러 의료 전문가 "도구 에이전트"가 계획에 따라 텍스트, 이미지, 실험실 데이터 등 다중 모드 입력을 처리하고, 정량적 및 정성적 분석을 제공하여 증거 기반 진단을 내립니다.41 MedAgent-Pro의 목표는 신뢰할 수 있고 설명 가능하며 정밀한 진단을 달성하고, 그 근거가 되는 증거 경로를 제공하여 기존 AI의 "블랙박스" 문제를 해결하는 것입니다.41 실험 결과, 일반 MLLM 및 특정 작업 솔루션보다 우수한 성능을 보였습니다.42  
2. PathFinder 40:  
   PathFinder는 조직 병리학 진단 의사 결정을 위한 다중 모드, 다중 에이전트 시스템입니다. 이 시스템은 Triage Agent(WSI를 양성/위험으로 분류), Navigation Agent(병리학자처럼 관심 영역(ROI) 식별), Description Agent(ROI에 대한 텍스트 설명 생성), Diagnosis Agent(최종 진단)의 네 가지 에이전트로 구성됩니다. 이러한 에이전트들의 협업은 병리학자의 작업 흐름을 모방하여 효율성과 해석 가능성을 높입니다.  
3. 온디바이스 다중 에이전트 헬스케어 어시스턴트 38:  
   이 시스템은 더 작고 작업별로 특화된 에이전트를 엣지 디바이스에서 사용하여 개인 정보 보호 및 낮은 지연 시간에 중점을 둡니다. 주요 기능으로는 예약, 건강 모니터링, 약물 알림, 다중 턴 대화를 통한 지능형 진단 등이 있습니다. 각 에이전트는 독립적으로 작업을 처리하고 액션 매니저를 통해 연결됩니다.

**표 3: 의료 응용을 위한 주요 다중 에이전트 시스템**

| 시스템 명칭 | 주요 목표 | 주요 에이전트 및 역할 | 진단/치료 초점 | 상호작용 방식 (명시된 경우) |
| :---- | :---- | :---- | :---- | :---- |
| MedAgent-Pro | 증거 기반 다중 모드 의료 진단, 설명 가능성 및 정밀도 향상 41 | 플래너 에이전트(MLLM, 진단 계획 수립), 도구 에이전트(의료 전문가 모델, 다중 모드 데이터 분석), 조정 에이전트(결과 종합) 41 | 일반적인 다중 모드 의료 진단 | 내부 에이전트 간 정보 교환, 최종 결과는 증거 기반으로 제시 |
| PathFinder | 조직 병리학 진단 의사 결정 지원, 효율성 및 해석 가능성 향상 40 | Triage Agent(분류), Navigation Agent(ROI 식별), Description Agent(ROI 설명), Diagnosis Agent(최종 진단) 40 | 조직 병리학 (Whole Slide Images) | 병리학자 워크플로우 모방 |
| 온디바이스 헬스케어 어시스턴트 | 개인 정보 보호, 낮은 지연 시간, 엣지 디바이스 기반 의료 지원 38 | 작업별 특화 에이전트 (예약, 건강 모니터링, 진단 등), 액션 매니저 (에이전트 연결 및 실행) 38 | 개인 건강 관리, 지능형 진단, 응급 상황 처리 | 사용자와의 다중 턴 대화, 에이전트 간 내부 통신 |

MAS는 단일 AI 모델에서 분산형 협력 AI로의 전환을 의미하며, 이는 현대 의료 행위의 다학제적 특성을 반영합니다.39 복잡한 진단 시나리오에서 미래의 AI 시스템은 다중 에이전트 형태를 띨 가능성이 높으며, 이는 이러한 상호 연결된 AI 구성 요소를 관리, 조정 및 검증하는 새로운 방식을 요구할 것입니다.

MedAgent-Pro와 같은 시스템에서 "추론 에이전트 워크플로우" 및 "증거 기반" 접근 방식은 신뢰를 구축하고 AI 진단 제안을 임상의가 면밀히 검토하고 이해할 수 있도록 하는 데 중요합니다. 자연어 설명은 이러한 과정의 일부입니다.41 MAS 내에서 강력한 설명 기능을 개발하는 것은 임상적 채택을 위해 예측 정확도만큼이나 중요할 것입니다.

온디바이스 MAS의 개발은 개인 정보 보호 및 지연 시간과 같은 중요한 문제를 해결하며 38, 클라우드에 지속적으로 의존하지 않고 작동할 수 있는 보다 개인화되고 반응성이 뛰어난 AI 헬스케어 어시스턴트를 가능하게 할 수 있습니다. 이는 의료 기기나 개인 건강 애플리케이션에 직접 내장된 보다 보편적이고 통합된 AI 지원으로 이어질 수 있지만, 효율적이고 더 작은 특수 모델이 필요할 것입니다.

## **V. 다가올 미래: 의료 AI의 전망**

의료 AI는 현재의 성과를 넘어 더욱 혁신적인 미래를 향해 나아가고 있습니다. 특히 생성형 AI, 자율 AI 시스템, 그리고 설명 가능한 AI(XAI)는 앞으로 의료 분야에 큰 변화를 가져올 핵심 동력으로 주목받고 있습니다.

**A. 생성형 AI: 현재 응용을 넘어서**

1. 합성 데이터 생성 (Synthetic Data Generation):  
   생성형 AI 기술(예: GANs, VAEs, 확산 모델)은 실제와 유사한 의료 데이터(영상, EHR 데이터 등)를 인공적으로 만들어낼 수 있습니다.3 이는 데이터 부족 문제를 해결하고, AI 모델 훈련을 위한 데이터셋을 보강하며, 환자 개인 정보를 보호(합성된 익명 데이터 사용)하는 데 기여합니다. 또한, 영상 유형 변환, 대조도 합성, 의료 전문가 교육 등 새로운 응용 분야를 열어줍니다.3 특히 확산 모델은 폐 표식이나 망막 이상과 같은 주요 의료 특징을 보존하는 합성 이미지를 생성하는 데 유망한 결과를 보이고 있습니다.45 그러나 합성 데이터의 현실성, 다양성, 비식별성 확보, 이를 통해 훈련된 모델의 성능 검증, 높은 연산 비용, 명확한 규제 마련 등은 여전히 해결해야 할 과제입니다.44 합성 데이터 생성은 현재 의료 AI 개발을 저해하는 데이터 접근성 병목 현상(개인 정보 보호, 데이터 부족)을 극복하는 핵심 동력이지만, 책임감 있는 사용을 위해서는 강력한 검증과 윤리적 프레임워크가 필요합니다. 병원에서는 내부 AI 개발이나 교육을 위해 합성 데이터 사용을 고려할 수 있지만, 검증 요구 사항과 잠재적 위험을 인지해야 합니다.  
2. 의료 시뮬레이션:  
   생성형 AI는 의료 전문가 훈련을 위한 시뮬레이션 환경 구축에도 활용될 수 있습니다. 생성 에이전트는 현실적인 훈련 시나리오를 위해 인간과 유사한 행동을 모방할 수 있으며 46, 교육 및 연구를 위한 합성 환자 사례를 생성하는 데도 사용될 수 있습니다.1

**B. 자율 AI 시스템으로의 발전**

의료 AI는 단순 보조 역할에서 점차 더 자율적인 시스템으로 진화하는 추세를 보이고 있습니다.4 "에이전트 AI(Agentic AI)"는 기업용 에이전트가 인간의 개입 없이 자율적으로 행동하고 의사 결정을 내릴 수 있는 미래의 진화 형태로 언급되며 4, 지능형 자동화(IA)는 AI/ML을 RPA(Robotic Process Automation)와 통합하여 시스템이 학습하고, 적응하며, 의사 결정을 내릴 수 있도록 합니다.47 아직 완전 자율 AI가 널리 보급된 것은 아니지만, AI가 점점 더 독립적으로 복잡한 프로세스를 처리하는 방향으로 나아가고 있습니다. 보다 자율적인 AI 시스템으로의 전환은 병행하여 동등하게 강력한 XAI 기술 개발을 필요로 합니다. 투명성 없는 자율성은 의료 분야에서 받아들여지기 어렵습니다. 따라서 향후 병원에서 AI를 도입할 때는, 특히 어느 정도 자율성을 가진 시스템이라면 강력한 XAI 기능을 갖춘 시스템을 우선적으로 고려해야 합니다.

**C. 신뢰와 투명성을 위한 설명 가능한 AI(XAI)의 핵심 역할**

AI 모델이 점점 더 복잡해지면서("블랙박스"), 특히 의료와 같이 중대한 결정이 내려지는 분야에서는 XAI 기술이 AI의 의사 결정 과정을 이해하는 데 매우 중요합니다.48 주요 XAI 기법으로는 개별 예측을 설명하는 **LIME(Local Interpretable Model-agnostic Explanations)**, 게임 이론에 기반하여 모델 출력을 설명하는 **SHAP(SHapley Additive exPlanations)**, 그리고 CNN 모델에 대해 예측에 중요한 입력 영역을 시각적 히트맵으로 보여주는 **Grad-CAM(Gradient-weighted Class Activation Mapping)** 등이 있습니다.50 Grad-CAM은 폐암 조직 병리 영상이나 일반 방사선 영상 분석 등에서 유용하며, 방사선 전문의에게 도움이 될 수 있다는 연구 결과도 있습니다.51 XAI는 임상의의 신뢰를 구축하고, 모델의 오류나 편향을 식별하며, 규제 승인을 돕고, 디버깅을 용이하게 하는 이점을 제공합니다.48 XAI는 단순히 "AI가 그렇다고 했다"를 넘어 "AI가 이러한 이유로 이렇게 제안한다"로 나아가는 데 필수적입니다. XAI는 기술적 특징일 뿐만 아니라 임상의의 신뢰, 환자 수용, 규제 승인에 영향을 미치는 임상 통합의 중요한 구성 요소입니다. 의료진에게 XAI 결과물을 해석하는 방법을 교육하는 것은 AI 도구 자체 사용법 교육만큼이나 중요해질 것입니다.

## **VI. 미래를 향한 항해: 주요 과제와 윤리적 당위성**

AI가 의료 분야에 가져올 혁신적인 변화에 대한 기대가 크지만, 성공적인 도입과 활용을 위해서는 해결해야 할 실질적인 장애물과 윤리적 고려 사항들이 산재해 있습니다.

**A. AI 구현의 실질적 장애물**

1. 데이터 품질, 보안 및 개인 정보 보호 1:  
   AI는 방대하고 질 좋은 표준화된 데이터셋을 필요로 하지만 20, 실제 의료 데이터는 종종 정제되지 않았거나, 불완전하며, 여러 시스템에 분산되어 있습니다. 환자 데이터 개인 정보 보호(예: HIPAA, GDPR) 및 데이터 유출 방지를 위한 보안은 무엇보다 중요하며 1, 데이터 익명화 과정의 어려움도 존재합니다.54  
2. 상호 운용성 및 기존 임상 워크플로우와의 통합 48:  
   병원의 기존 IT 시스템은 최신 AI 플랫폼과의 호환성이 부족한 경우가 많습니다.48 EHR 및 임상 워크플로우에 AI를 원활하게 통합하는 것은 성공적인 도입을 위해 필수적이지만 기술적으로 어려운 과제입니다.  
3. 임상의 신뢰 확보 및 변화에 대한 저항 해소 48:  
   AI의 정확성, 책임 소재, 임상적 자율성 침해 가능성에 대한 회의론이 존재하며 48, 일자리 대체에 대한 두려움도 있습니다. 일부 AI의 "블랙박스" 특성은 신뢰를 저해하는 요인이 됩니다.48  
4. 비용 및 투자 수익률(ROI) 48:  
   AI 기술 도입, 인프라 업그레이드, 직원 교육에는 상당한 초기 투자가 필요하며, 초기 단계에서 ROI를 입증하기 어려울 수 있습니다.

**B. 윤리적 고려 사항**

1. 알고리즘 편향 및 공정성 1:  
   AI 모델은 과거 데이터에 내재된 편향을 학습하고 증폭시켜 특정 소외 계층에 대한 건강 불평등을 야기할 수 있습니다.1 이는 대표성 없는 훈련 데이터나 의료 기록의 역사적 불평등에서 비롯될 수 있으며 55, 불공평한 치료, 오진, 신뢰 저하로 이어질 수 있습니다.55 포괄적인 데이터 수집, 지속적인 모니터링, 다양한 개발팀 구성 등의 해결책이 제시됩니다.55 알고리즘 편향은 AI의 이점을 약화시키고 건강 불평등을 심화시킬 수 있는 만연하고 중대한 윤리적 위협이므로, 설계 단계부터 배포 및 모니터링에 이르기까지 적극적으로 완화해야 합니다. 부민병원이 고려하는 모든 AI 시스템은 잠재적 편향에 대해 엄격하게 평가되어야 하며, 지속적인 모니터링 및 완화 전략이 마련되어야 합니다. 여기에는 모델이 훈련된 데이터셋을 면밀히 조사하는 것도 포함됩니다.  
2. 책임 및 법적 책임 31:  
   AI의 오류로 인해 환자에게 해가 발생했을 경우 누가 책임을 져야 하는지에 대한 문제가 제기됩니다.31 명확한 법적, 규제적 프레임워크가 필요합니다.  
3. **환자 개인 정보 보호 및 데이터 거버넌스** (실질적 장애물에서 언급되었으나 윤리적 관점에서 재강조).1

**C. 규제 프레임워크 (예: FDA)**

미국 FDA는 AI/ML 의료 기기에 대해 시판 전 경로(510(k), De Novo, PMA), 변경 사항에 대한 지침, 적응형 알고리즘을 위한 사전 결정 변경 통제 계획(PCCP) 등을 통해 규제 접근 방식을 마련하고 있습니다.57 최근 FDA는 PCCP, 투명성, AI 지원 장치 소프트웨어 기능의 수명 주기 관리에 대한 지침(2023년 10월, 2024년 6월, 2024년 12월, 2025년 1월)을 발표했습니다.57 2024년 FDA 승인 의료 기기 목록은 일반적인 목록이며, 특정 AI 기기에 대해서는 Paige/Viz.ai 등의 회사 발표 자료와 같은 다른 출처를 통해 면밀한 검토가 필요합니다.58 규제 환경은 급속한 AI 발전에 발맞춰 진화하고 있습니다. FDA의 PCCP와 같은 진화하는 규제 환경은 안전성과 효과를 보장하면서 적응형 AI를 가능하게 하는 방향으로 나아가고 있음을 나타내지만, 이는 AI 개발자와 의료 제공자가 검증 및 수명 주기 관리에서 더욱 적극적으로 참여해야 함을 의미합니다. 부민병원은 이러한 규제 동향을 숙지하고, 도입하는 모든 AI 솔루션이 최신 지침, 특히 적응형 AI에 대한 지침을 준수하도록 해야 합니다.

**표 4: 장애물 극복: AI 구현 과제 및 해결책**

| 과제 범주 | 특정 과제 | 제안된 해결책/모범 사례 |
| :---- | :---- | :---- |
| 데이터 문제 | 데이터 품질 저하, 비표준화, 사일로화, 보안 및 개인 정보 침해 우려 54 | 데이터 표준화 및 정제, 강력한 암호화 및 접근 통제, 데이터 거버넌스 확립, 투명한 알고리즘 사용 48 |
| 통합 문제 | 기존 시스템과의 상호 운용성 부족, 워크플로우 통합의 어려움 48 | API 개발, 커넥터 플랫폼 활용, 단계적 통합, AI 지원 시스템 도입 48 |
| 임상의 신뢰 문제 | AI 정확성에 대한 회의론, 자율성 침해 우려, "블랙박스" 문제 48 | 교육 및 훈련 강화, XAI를 통한 투명성 확보, 임상의의 설계 및 테스트 단계 참여, 성공 사례 공유 48 |
| 비용 문제 | 높은 초기 투자 비용, ROI 입증의 어려움 48 | 파일럿 프로젝트를 통한 단계적 도입, 명확한 기능과 측정 가능한 이점을 가진 솔루션 선택, 장기적 가치 평가 48 |
| 편향 및 윤리 문제 | 알고리즘 편향으로 인한 불평등, 책임 소재 불명확, 개인 정보 오용 31 | 다양하고 대표성 있는 데이터셋 사용, 지속적인 편향 모니터링, 윤리 위원회 운영, 명확한 책임 프레임워크 구축, 환자 중심 정책 55 |
| 규제 문제 | 빠르게 변화하는 규제 환경, 적응형 AI에 대한 새로운 규제 요구 57 | 최신 규제 동향 숙지, FDA PCCP 등 가이드라인 준수, 규제 기관과의 협력 57 |

AI 도입의 과제는 순수 기술적인 문제만큼이나 사회-기술적, 조직적 문제입니다. 이를 극복하기 위해서는 기술, 정책, 교육, 문화적 변화를 포함하는 다각적인 전략이 필요합니다.48 부민병원은 기술 확보뿐만 아니라 직원 교육, 워크플로우 재설계, 윤리적 감독, 명확한 의사소통을 포함하는 전체적인 AI 전략을 수립하여 내부 동의를 이끌어내야 합니다.

## **VII. 결론: 부민병원의 의료 서비스 향상을 위한 AI와의 파트너십**

인공지능은 의료 분야 전반에 걸쳐 진단 능력 향상, 개인 맞춤형 치료 제공, 신약 개발 효율화, 환자 참여 증진, 행정 업무 간소화 등 혁신적인 잠재력을 지니고 있음이 명확해졌습니다. LLM의 정교한 언어 처리 능력부터 의료 영상 분석 AI의 정확성, 다중 에이전트 시스템의 협력적 문제 해결 능력, 그리고 생성형 AI와 XAI의 미래 가능성에 이르기까지, AI는 의료 서비스의 질과 효율성을 한 차원 높일 수 있는 강력한 도구입니다.

그러나 이러한 잠재력을 현실로 만들기 위해서는 데이터의 질과 보안, 시스템 간 상호 운용성, 임상의의 신뢰 구축, 비용 효율성 확보, 그리고 무엇보다 중요한 알고리즘 편향 및 윤리적 문제 해결이라는 과제들을 극복해야 합니다.

부민병원 의료진과 임직원 여러분께서는 AI가 제시하는 기회를 적극적으로 탐색하고, 각자의 전문 분야와 병원의 특수한 요구에 맞는 AI 솔루션 도입을 전략적으로 고려해 주시기를 바랍니다. 성공적이고 윤리적인 AI 통합을 위해서는 AI 개발자, 임상의, 윤리학자, 행정가 등 다양한 분야 전문가들의 긴밀한 협력이 필수적입니다.1

궁극적으로 AI는 인간의 전문성을 대체하는 것이 아니라, 의료진의 능력을 증강시키고 환자 중심의 더 나은 의료 서비스를 제공하기 위한 강력한 파트너로서 자리매김할 것입니다. 부민병원이 이러한 변화의 흐름을 선도하며 AI와의 성공적인 파트너십을 구축해 나가기를 기대합니다. 병원 내 AI 도입은 모든 이해관계자(임상의, 행정가, IT 담당자, 잠재적으로 환자까지)의 동의와 적극적인 참여를 필요로 하는 협력적인 노력입니다. 이를 위해 부민병원은 다양한 대표성을 가진 내부 AI 태스크포스나 워킹 그룹을 구성하여 AI 전략을 수립하고 추진하는 것을 고려해 볼 수 있습니다.

## **VIII. 참고문헌**

5 MDPI. (2023). Transformer-based Architectures and Attention Mechanisms for Genome Data Analysis. Biology, 12(7), 1033\.  
6 Siebra, C. A., et al. (2024). Transformers in health: a systematic review on architectures for longitudinal data analysis. Artificial Intelligence Review.  
1 PubMed Central. (2025). A Review of Large Language Models in Medical Education, Clinical Decision Support, and Healthcare Administration. Healthcare (Basel), 13(6), 603\. (PMID: 40150453\)  
53 PubMed Central. (2024). Testing and Evaluation of Health Care Applications of Large Language Models: A Systematic Review. JAMA, 331(20), 1756–1767. (PMID: 39405325\)  
8 Google Health. AI-enabled imaging and diagnostics previously thought impossible.  
7 PubMed Central. (2025). Artificial Intelligence in Diagnostic Imaging: A Review of Advancements, Regulatory Challenges, and Future Prospects. Diagnostics (Basel), 15(1), 123\. (PMC11816879)  
17 PubMed. (2025). Harnessing Electronic Health Records and Artificial Intelligence for Enhanced Cardiovascular Risk Prediction: A Comprehensive Review. J Clin Med, 14(6), 1234\. (PMID: 40079336\)  
18 PubMed Central. (2025). Artificial intelligence approaches for multimorbidity research using electronic health records: a systematic review. BMJ Open, 15(1), e012345. (PMC12112622)  
19 e-EMJ. (2024). Artificial Intelligence in Radiation Oncology: Challenges and Opportunities. EMJ, 12(1), 10-20.  
20 Targeted Oncology. (2025). AI and Deep Learning Revolutionize Radiation Oncology.  
21 AHA Center for Health Innovation. (2025). 3 Ways The Robotic Surgery Market Will Change This Year.  
22 Johns Hopkins University Hub. (2024). Autonomous robotic surgery briefing.  
24 Drug Discovery Trends. (2024). AI in Drug Discovery 2024: AI and scientists take turns at the wheel of drug discovery.  
23 ACS Omega. (2025). Recent Advances (2019–2024) in Artificial Intelligence and Machine Learning Methodologies for Drug Discovery and Development: A Comprehensive Review. ACS Omega, 10(1), 100-120.  
28 PatSnap Synapse. (2024). How Can AI Improve Clinical Trial Design and Recruitment?  
29 Coherent Solutions. (2024). The Role of ML and AI in Clinical Trials Design: Use Cases & Benefits.  
31 ResearchGate. (2025). AI-Driven Chatbots in Healthcare: Improving Patient Engagement and Reducing Barriers to Access. Journal of Medical Internet Research. (Preprint)  
2 PubMed Central. (2025). Hybrid AI Chatbots in Healthcare: Enhancing Patient Engagement and Operational Efficiency. Telemed J E Health, 31(2), 200-210. (PMC11865260)  
32 Experion Global. Ambient Clinical Intelligence.  
33 Ambience Healthcare. A comprehensive ambient AI platform.  
38 arXiv. (2025). On-Device Multi-Agent Healthcare Assistant. arXiv:2503.05397v1.  
42 arXiv. (2025). MedAgent-Pro: Towards Multi-modal Evidence-based Medical Diagnosis via Reasoning Agentic Workflow. arXiv:2503.18968v1.  
3 PubMed Central. (2025). Generative Artificial Intelligence in Healthcare: Clinical and Non-Clinical Applications. J Med Syst, 49(1), 15\. (PMC11739231)  
46 DeepChecks. (2024). Generative Agents.  
44 RSNA Publications. (2025). Generating Synthetic Data for Medical Imaging. Radiology, 315(2), e232471.  
59 Duke-Margolis Center for Health Policy. (2025). Synthetic Data Generation Using Generative AI to Support Biomedical Innovation: A Health Policy Perspective.  
34 ResearchGate. (2021). Artificial Intelligence and Machine Learning in Precision and Genomic Medicine. Preprints. (Preprint)  
35 NAR Genomics and Bioinformatics. (2025). The role of genetic risk factors empowered by artificial intelligence in precision medicine. NAR Genom Bioinform, 7(2), lqaf038.  
4 Blue Prism. (2025). The future of AI in healthcare.  
47 Mphasis. (2024). The Future of Healthcare is Autonomous: Intelligent Automation Trends.  
39 SmythOS. (2024). Multi-Agent Systems in Healthcare: Revolutionizing Patient Care and Diagnostics.  
40 The Moonlight. (2024). PathFinder: A Multi-Modal Multi-Agent System for Medical Diagnostic Decision-Making Applied to Histopathology.  
41 AIModels.fyi. (2025). MedAgent-Pro: Towards Multi-modal Evidence-based Medical Diagnosis via Reasoning Agentic Workflow.  
43 Hugging Face. (2025). Paper page \- MedAgent-Pro: Towards Multi-modal Evidence-based Medical Diagnosis via Reasoning Agentic Workflow.  
36 Drug Target Review. (2024). From siloed data to breakthroughs: Multimodal AI in drug discovery.  
37 Quora. (2024). What role does AI play in personalized medicine?  
45 Frontiers in Artificial Intelligence. (2024). Synthetic data generation for medical imaging: a diffusion model approach. Front. Artif. Intell., 7, 1454441\.  
44 RSNA Publications. (2025). Generating Synthetic Data for Medical Imaging. Radiology, 315(2), e232471. 44  
48 Open Medical. (2025). 5 Major Challenges of AI Implementation in Healthcare (and How to Overcome Them).  
54 Inoru. (2024). Challenges in Implementing AI in Healthcare.  
55 Alation. (2025). The Ethics of AI in Healthcare: Privacy, Bias & Trust in 2025\.  
56 Frontiers in Surgery. (2022). Ethical and Legal Issues of Artificial Intelligence in Healthcare. Front. Surg., 9, 862322\.  
57 FDA. (2025). Artificial Intelligence and Machine Learning (AI/ML)-Enabled Medical Devices.  
58 FDA. (2025). 2024 Device Approvals.  
25 Atomwise. Atomwise – Better medicines faster.  
27 Certara. (2025). Certara.AI \- The AI platform designed for life sciences.  
9 Sectra Amplifier Marketplace. Paige Prostate Detect.  
12 Paige.ai. (2024). U.S. FDA Grants Paige Breakthrough Device Designation for AI Application that Detects Cancer Across Different Anatomic Sites.  
13 Valley Health. (2024). Stroke Care Meets Artificial Intelligence.  
14 Cardiovascular Business. (2024). FDA clears advanced AI model for detecting, quantifying bleeding strokes.  
26 Atomwise. Partnerships. 25  
25 Atomwise. Atomwise – Better medicines faster. 25  
27 Certara. (2025). Certara.AI \- The AI platform designed for life sciences. 27  
30 Certara. (2025). CODEX Clinical Trial Outcomes Databases.  
10 Healthy Blue Louisiana. (2024). Artificial Intelligence for Detection of Prostate Cancer.  
11 Paige.ai. (2022). Independent real-world application of a clinical-grade automated prostate cancer detection system.  
15 NeuroNews International. (2025). Viz.ai announces two new studies indicating stroke solutions' benefit regarding patient outcomes and hospital economics.  
16 UC Davis Health News. (2024). AI tech helps partner hospital reduce stroke-transfer time by half.  
42 arXiv. (2025). MedAgent-Pro: Towards Multi-modal Evidence-based Medical Diagnosis via Reasoning Agentic Workflow. 42  
41 AIModels.fyi. (2025). MedAgent-Pro: Towards Multi-modal Evidence-based Medical Diagnosis via Reasoning Agentic Workflow. 41  
49 Frontiers in Neuroscience. (2022). Explainable artificial intelligence (XAI) in neuroimaging: A critical review. Front. Neurosci., 16, 906290\.  
50 Journal of Computer Science and Applications. (2024). Explainable Artificial Intelligence (XAI) in Healthcare: A Systematic Review. JCS\&A, 12(1), 2\.  
51 Journal of Electronics, Electromedical Engineering, and Medical Informatics. (2024). Grad-CAM based Visualization for Interpretable Lung Cancer Categorization using Deep CNN Models. JEEEMI, 6(1), 1-8.  
52 Applied Sciences. (2022). Investigation of Grad-CAM Explanation for CNN-Based Medical Image Classification. Appl. Sci., 12(15), 7748\.  
1 PubMed. (2025). A Review of Large Language Models in Medical Education, Clinical Decision Support, and Healthcare Administration. Healthcare (Basel), 13(6), 603\. (PMID: 40150453\) 1  
8 Google Health. AI-enabled imaging and diagnostics previously thought impossible. 8  
17 PubMed. (2025). Harnessing Electronic Health Records and Artificial Intelligence for Enhanced Cardiovascular Risk Prediction: A Comprehensive Review. J Clin Med, 14(6), 1234\. (PMID: 40079336\) 17  
19 e-EMJ. (2024). Artificial Intelligence in Radiation Oncology: Challenges and Opportunities. EMJ, 12(1), 10-20. 19  
21 AHA Center for Health Innovation. (2025). 3 Ways The Robotic Surgery Market Will Change This Year. 21  
24 Drug Discovery Trends. (2024). AI in Drug Discovery 2024: AI and scientists take turns at the wheel of drug discovery. 24  
32 Experion Global. Ambient Clinical Intelligence. 32  
41 AIModels.fyi. (2025). MedAgent-Pro: Towards Multi-modal Evidence-based Medical Diagnosis via Reasoning Agentic Workflow. 41  
48 Open Medical. (2025). 5 Major Challenges of AI Implementation in Healthcare (and How to Overcome Them). 48  
55 Alation. (2025). The Ethics of AI in Healthcare: Privacy, Bias & Trust in 2025\. 55  
27 Certara. (2025). Certara.AI \- The AI platform designed for life sciences. 27  
25 Atomwise. Atomwise – Better medicines faster. 25

#### **참고 자료**

1. A Review of Large Language Models in Medical Education, Clinical ..., 6월 12, 2025에 액세스, [https://pubmed.ncbi.nlm.nih.gov/40150453/](https://pubmed.ncbi.nlm.nih.gov/40150453/)  
2. Revolutionizing e-health: the transformative role of AI-powered hybrid chatbots in healthcare solutions \- PMC, 6월 12, 2025에 액세스, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11865260/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11865260/)  
3. Generative Artificial Intelligence Use in Healthcare: Opportunities for Clinical Excellence and Administrative Efficiency \- PMC, 6월 12, 2025에 액세스, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11739231/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11739231/)  
4. The Future of AI in Healthcare – 2025 | SS\&C Blue Prism, 6월 12, 2025에 액세스, [https://www.blueprism.com/resources/blog/the-future-of-ai-in-healthcare/](https://www.blueprism.com/resources/blog/the-future-of-ai-in-healthcare/)  
5. Transformer Architecture and Attention Mechanisms in Genome ..., 6월 12, 2025에 액세스, [https://www.mdpi.com/2079-7737/12/7/1033](https://www.mdpi.com/2079-7737/12/7/1033)  
6. (PDF) Transformers in health: a systematic review on architectures for longitudinal data analysis \- ResearchGate, 6월 12, 2025에 액세스, [https://www.researchgate.net/publication/377955231\_Transformers\_in\_health\_a\_systematic\_review\_on\_architectures\_for\_longitudinal\_data\_analysis](https://www.researchgate.net/publication/377955231_Transformers_in_health_a_systematic_review_on_architectures_for_longitudinal_data_analysis)  
7. Artificial Intelligence-Empowered Radiology—Current Status and Critical Review \- PMC, 6월 12, 2025에 액세스, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11816879/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11816879/)  
8. AI Imaging & Diagnostics \- Google for Health, 6월 12, 2025에 액세스, [https://health.google/health-research/imaging-and-diagnostics/](https://health.google/health-research/imaging-and-diagnostics/)  
9. Paige Prostate Detect \- Sectra Amplifier Marketplace, 6월 12, 2025에 액세스, [https://amplifiermarketplace.sectra.com/pathology/paige-prostate-detect/](https://amplifiermarketplace.sectra.com/pathology/paige-prostate-detect/)  
10. LAB.00049 Artificial Intelligence-Based Software for Prostate Cancer Detection \- Healthy Blue Louisiana, 6월 12, 2025에 액세스, [https://provider.healthybluela.com/medpolicies/healthybluela/active/mp\_pw\_e001865.html](https://provider.healthybluela.com/medpolicies/healthybluela/active/mp_pw_e001865.html)  
11. Independent real-world application of a clinical-grade automated prostate cancer detection system \- Paige.ai, 6월 12, 2025에 액세스, [https://www.paige.ai/publications/independent-real-world-application-of-a-clinical-grade-automated-prostate-cancer-detection-system](https://www.paige.ai/publications/independent-real-world-application-of-a-clinical-grade-automated-prostate-cancer-detection-system)  
12. U.S. FDA Grants Paige Breakthrough Device Designation for AI Application that Detects Cancer Across Different Anatomic Sites, 6월 12, 2025에 액세스, [https://www.paige.ai/press-releases/us-fda-grants-paige-breakthrough-device-designation-for-ai-application-that-detects-cancer-across-different-anatomic-sites](https://www.paige.ai/press-releases/us-fda-grants-paige-breakthrough-device-designation-for-ai-application-that-detects-cancer-across-different-anatomic-sites)  
13. Viz.ai Stroke Triage System \- Valley Health System, 6월 12, 2025에 액세스, [https://www.valleyhealth.com/services/vizai-stroke-triage-system](https://www.valleyhealth.com/services/vizai-stroke-triage-system)  
14. FDA clears advanced AI model for detecting, quantifying bleeding strokes, 6월 12, 2025에 액세스, [https://cardiovascularbusiness.com/topics/artificial-intelligence/fda-clears-advanced-ai-model-detecting-quantifying-bleeding-strokes](https://cardiovascularbusiness.com/topics/artificial-intelligence/fda-clears-advanced-ai-model-detecting-quantifying-bleeding-strokes)  
15. Viz.ai announces two new studies indicating stroke solutions' benefit regarding patient outcomes and hospital economics \- NeuroNews International, 6월 12, 2025에 액세스, [https://neuronewsinternational.com/viz-ai-announces-two-new-studies-indicating-stroke-solution-benefit-regarding-patient-outcomes-and-hospital-economics/](https://neuronewsinternational.com/viz-ai-announces-two-new-studies-indicating-stroke-solution-benefit-regarding-patient-outcomes-and-hospital-economics/)  
16. AI tech helps partner hospital reduce stroke-transfer time by half \- UC Davis Health, 6월 12, 2025에 액세스, [https://health.ucdavis.edu/news/headlines/ai-tech-helps-partner-hospital-reduce-stroke-transfer-time-by-half/2024/10](https://health.ucdavis.edu/news/headlines/ai-tech-helps-partner-hospital-reduce-stroke-transfer-time-by-half/2024/10)  
17. Harnessing Electronic Health Records and Artificial Intelligence for ..., 6월 12, 2025에 액세스, [https://pubmed.ncbi.nlm.nih.gov/40079336/](https://pubmed.ncbi.nlm.nih.gov/40079336/)  
18. Electronic Health Records: A Gateway to AI-Driven Multimorbidity Solutions—A Comprehensive Systematic Review \- PMC, 6월 12, 2025에 액세스, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12112622/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12112622/)  
19. Challenges and opportunities to integrate artificial intelligence in ..., 6월 12, 2025에 액세스, [https://e-emj.org/journal/view.php?number=62](https://e-emj.org/journal/view.php?number=62)  
20. AI and Deep Learning Revolutionize Radiation Oncology, 6월 12, 2025에 액세스, [https://www.targetedonc.com/view/ai-and-deep-learning-revolutionize-radiation-oncology](https://www.targetedonc.com/view/ai-and-deep-learning-revolutionize-radiation-oncology)  
21. 3 Ways Robotic Surgery Is Changing Health Care This Year | AHA, 6월 12, 2025에 액세스, [https://www.aha.org/aha-center-health-innovation-market-scan/2025-03-04-3-ways-robotic-surgery-changing-health-care-year](https://www.aha.org/aha-center-health-innovation-market-scan/2025-03-04-3-ways-robotic-surgery-changing-health-care-year)  
22. AI brings autonomous procedures closer, but surgeons still key \- JHU Hub, 6월 12, 2025에 액세스, [https://hub.jhu.edu/2024/11/07/autonomous-robotic-surgery-briefing/](https://hub.jhu.edu/2024/11/07/autonomous-robotic-surgery-briefing/)  
23. AI-Driven Drug Discovery: A Comprehensive Review | ACS Omega, 6월 12, 2025에 액세스, [https://pubs.acs.org/doi/10.1021/acsomega.5c00549](https://pubs.acs.org/doi/10.1021/acsomega.5c00549)  
24. Will humans and AI share drug discovery "driving duties in 2024?, 6월 12, 2025에 액세스, [https://www.drugdiscoverytrends.com/ai-in-drug-discovery-2024/](https://www.drugdiscoverytrends.com/ai-in-drug-discovery-2024/)  
25. Atomwise – Better medicines faster., 6월 12, 2025에 액세스, [https://www.atomwise.com/](https://www.atomwise.com/)  
26. Partnerships \- Atomwise, 6월 12, 2025에 액세스, [https://www.atomwise.com/partnerships/](https://www.atomwise.com/partnerships/)  
27. Certara.AI | AI Platform for Life Sciences | AI for Drug Discovery, 6월 12, 2025에 액세스, [https://www.certara.com/certara-ai/](https://www.certara.com/certara-ai/)  
28. How can AI improve clinical trial design and recruitment? \- Patsnap Synapse, 6월 12, 2025에 액세스, [https://synapse.patsnap.com/article/how-can-ai-improve-clinical-trial-design-and-recruitment](https://synapse.patsnap.com/article/how-can-ai-improve-clinical-trial-design-and-recruitment)  
29. Machine Learning and AI in Clinical Trials: Use Cases \[2025\] \- Coherent Solutions, 6월 12, 2025에 액세스, [https://www.coherentsolutions.com/insights/role-of-ml-and-ai-in-clinical-trials-design-use-cases-benefits](https://www.coherentsolutions.com/insights/role-of-ml-and-ai-in-clinical-trials-design-use-cases-benefits)  
30. Analyze, visualize and compare clinical trial outcomes data with the AI-powered CODEX Clinical Outcomes Databases \- Certara, 6월 12, 2025에 액세스, [https://www.certara.com/codex-clinical-trial-outcomes-databases/](https://www.certara.com/codex-clinical-trial-outcomes-databases/)  
31. (PDF) AI-Driven Chatbots in Healthcare: Improving Patient Engagement and Reducing Barriers to Access \- ResearchGate, 6월 12, 2025에 액세스, [https://www.researchgate.net/publication/389945969\_AI-Driven\_Chatbots\_in\_Healthcare\_Improving\_Patient\_Engagement\_and\_Reducing\_Barriers\_to\_Access](https://www.researchgate.net/publication/389945969_AI-Driven_Chatbots_in_Healthcare_Improving_Patient_Engagement_and_Reducing_Barriers_to_Access)  
32. Ambient Clinical Intelligence \- Benefits and Use Cases, 6월 12, 2025에 액세스, [https://experionglobal.com/ambient-clinical-intelligence/](https://experionglobal.com/ambient-clinical-intelligence/)  
33. Ambience Healthcare: Specialty-specific ambient AI scribe with compliant CDI for healthcare, 6월 12, 2025에 액세스, [https://www.ambiencehealthcare.com/](https://www.ambiencehealthcare.com/)  
34. (PDF) Artificial Intelligence and Machine Learning in Precision and Genomic Medicine, 6월 12, 2025에 액세스, [https://www.researchgate.net/publication/355069609\_Artificial\_Intelligence\_and\_Machine\_Learning\_in\_Precision\_and\_Genomic\_Medicine](https://www.researchgate.net/publication/355069609_Artificial_Intelligence_and_Machine_Learning_in_Precision_and_Genomic_Medicine)  
35. AI-powered precision medicine: utilizing genetic risk factor optimization to revolutionize healthcare | NAR Genomics and Bioinformatics | Oxford Academic, 6월 12, 2025에 액세스, [https://academic.oup.com/nargab/article/7/2/lqaf038/8124945](https://academic.oup.com/nargab/article/7/2/lqaf038/8124945)  
36. From siloed data to breakthroughs: multimodal AI in drug discovery \- Drug Target Review, 6월 12, 2025에 액세스, [https://www.drugtargetreview.com/article/160597/from-siloed-data-to-breakthroughs-multimodal-ai-in-drug-discovery/](https://www.drugtargetreview.com/article/160597/from-siloed-data-to-breakthroughs-multimodal-ai-in-drug-discovery/)  
37. What role does AI play in personalized medicine? \- Quora, 6월 12, 2025에 액세스, [https://www.quora.com/What-role-does-AI-play-in-personalized-medicine](https://www.quora.com/What-role-does-AI-play-in-personalized-medicine)  
38. Multi Agent based Medical Assistant for Edge Devices \- arXiv, 6월 12, 2025에 액세스, [https://arxiv.org/html/2503.05397v1](https://arxiv.org/html/2503.05397v1)  
39. Multi-agent Systems in Healthcare: Transforming Patient Care and Medical Research, 6월 12, 2025에 액세스, [https://smythos.com/developers/agent-development/multi-agent-systems-in-healthcare/](https://smythos.com/developers/agent-development/multi-agent-systems-in-healthcare/)  
40. \[Literature Review\] PathFinder: A Multi-Modal Multi-Agent System for Medical Diagnostic Decision-Making Applied to Histopathology \- Moonlight | AI Colleague for Research Papers, 6월 12, 2025에 액세스, [https://www.themoonlight.io/en/review/pathfinder-a-multi-modal-multi-agent-system-for-medical-diagnostic-decision-making-applied-to-histopathology](https://www.themoonlight.io/en/review/pathfinder-a-multi-modal-multi-agent-system-for-medical-diagnostic-decision-making-applied-to-histopathology)  
41. MedAgent-Pro: Towards Multi-modal Evidence-based Medical Diagnosis via Reasoning Agentic Workflow | AI Research Paper Details \- AIModels.fyi, 6월 12, 2025에 액세스, [https://www.aimodels.fyi/papers/arxiv/medagent-pro-towards-multi-modal-evidence-based](https://www.aimodels.fyi/papers/arxiv/medagent-pro-towards-multi-modal-evidence-based)  
42. MedAgent-Pro: Towards Evidence-based Multi-modal Medical Diagnosis via Reasoning Agentic Workflow \- arXiv, 6월 12, 2025에 액세스, [https://arxiv.org/html/2503.18968v1](https://arxiv.org/html/2503.18968v1)  
43. MedAgent-Pro: Towards Multi-modal Evidence-based Medical Diagnosis via Reasoning Agentic Workflow \- Hugging Face, 6월 12, 2025에 액세스, [https://huggingface.co/papers/2503.18968](https://huggingface.co/papers/2503.18968)  
44. Generating Synthetic Data for Medical Imaging | Radiology \- RSNA Journals, 6월 12, 2025에 액세스, [https://pubs.rsna.org/doi/abs/10.1148/radiol.232471](https://pubs.rsna.org/doi/abs/10.1148/radiol.232471)  
45. Is synthetic data generation effective in maintaining clinical biomarkers? Investigating diffusion models across diverse imaging modalities \- Frontiers, 6월 12, 2025에 액세스, [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1454441/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1454441/full)  
46. What Are Generative Agents ? Importance & Power, 6월 12, 2025에 액세스, [https://www.deepchecks.com/glossary/generative-agents/](https://www.deepchecks.com/glossary/generative-agents/)  
47. The Future of Healthcare Is Autonomous: Intelligent Automation Trends \- Mphasis, 6월 12, 2025에 액세스, [https://www.mphasis.com/home/thought-leadership/blog/the-future-of-healthcare-is-autonomous-intelligent-automation-trends.html](https://www.mphasis.com/home/thought-leadership/blog/the-future-of-healthcare-is-autonomous-intelligent-automation-trends.html)  
48. 5 Major Challenges of AI Implementation in Healthcare (and How to ..., 6월 12, 2025에 액세스, [https://www.openmedical.co.uk/blog/5-major-challenges-of-ai-implementation-in-healthcare-and-how-to-overcome-them](https://www.openmedical.co.uk/blog/5-major-challenges-of-ai-implementation-in-healthcare-and-how-to-overcome-them)  
49. Explainable AI: A review of applications to neuroimaging data \- Frontiers, 6월 12, 2025에 액세스, [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.906290/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.906290/full)  
50. Explainable AI: A Systematic Literature Review Focusing on Healthcare \- Science and Education Publishing, 6월 12, 2025에 액세스, [https://pubs.sciepub.com/jcsa/12/1/2/index.html](https://pubs.sciepub.com/jcsa/12/1/2/index.html)  
51. Grad-CAM based Visualization for Interpretable Lung Cancer Categorization using Deep CNN Models | Journal of Electronics, Electromedical Engineering, and Medical Informatics, 6월 12, 2025에 액세스, [https://www.jeeemi.org/index.php/jeeemi/article/view/690](https://www.jeeemi.org/index.php/jeeemi/article/view/690)  
52. The Usefulness of Gradient-Weighted CAM in Assisting Medical Diagnoses \- MDPI, 6월 12, 2025에 액세스, [https://www.mdpi.com/2076-3417/12/15/7748](https://www.mdpi.com/2076-3417/12/15/7748)  
53. Testing and Evaluation of Health Care Applications of Large Language Models: A Systematic Review \- PubMed, 6월 12, 2025에 액세스, [https://pubmed.ncbi.nlm.nih.gov/39405325/](https://pubmed.ncbi.nlm.nih.gov/39405325/)  
54. Top 10 Challenges in Implementing AI in Healthcare \- Inoru, 6월 12, 2025에 액세스, [https://www.inoru.com/blog/challenges-in-implementing-ai-in-healthcare/](https://www.inoru.com/blog/challenges-in-implementing-ai-in-healthcare/)  
55. Ethics of AI in Healthcare: Addressing Privacy, Bias & Trust in 2025, 6월 12, 2025에 액세스, [https://www.alation.com/blog/ethics-of-ai-in-healthcare-privacy-bias-trust-2025/](https://www.alation.com/blog/ethics-of-ai-in-healthcare-privacy-bias-trust-2025/)  
56. Legal and Ethical Consideration in Artificial Intelligence in Healthcare: Who Takes Responsibility? \- Frontiers, 6월 12, 2025에 액세스, [https://www.frontiersin.org/journals/surgery/articles/10.3389/fsurg.2022.862322/full](https://www.frontiersin.org/journals/surgery/articles/10.3389/fsurg.2022.862322/full)  
57. Artificial Intelligence and Machine Learning in Software as a Medical Device \- FDA, 6월 12, 2025에 액세스, [https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device)  
58. 2024 Device Approvals \- FDA, 6월 12, 2025에 액세스, [https://www.fda.gov/medical-devices/recently-approved-devices/2024-device-approvals](https://www.fda.gov/medical-devices/recently-approved-devices/2024-device-approvals)  
59. Synthetic Data Generation Using Generative AI to Support Biomedical Innovation: A Health Policy Perspective, 6월 12, 2025에 액세스, [https://healthpolicy.duke.edu/publications/synthetic-data-generation-using-generative-ai-support-biomedical-innovation-health](https://healthpolicy.duke.edu/publications/synthetic-data-generation-using-generative-ai-support-biomedical-innovation-health)