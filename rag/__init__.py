"""강아지 RAG 저장·검색 — 영상 분석 산출물을 입양 응대봇 근거로.

설계(Notion '강아지 RAG 저장·검색 — 최종 설계', 07-08 개정) 요지:
  · 3종 데이터 — A 정형 카드(임베딩 없이 통째 주입) / B L1 행동 프로필(trait 요약)
    / C L2 에피소드(영상당 관찰 서술 + M4 타임스탬프).
  · 저장 위치별 텍스트 출처 고정 — 환각 위험은 conf 가 아니라 그 문장을 누가
    만들었느냐에 달렸다. 출처 컬럼 없이 테이블·layer 단위로 출처를 통일한다:
    카드=사람 입력 / L1=LLM 생성(유일한 생성 지점, 주장 분해 사실 검수 필수) /
    L2=관찰 문장 원문(코드 조립만, LLM 수정 금지). 저작(author) 텍스트는 DB 에
    넣지 않는다.
  · re-ID 확정 게이트 — foster 확정된 잡의 산출물만 인제스트(meta.foster_uncertain
    이 비어 있어야). 틀린 re-ID 는 다른 개의 행동을 이 개 프로필에 조용히 기록한다.
  · 스택 — pgvector(Postgres) 단일, bge-m3(1024차원) 임베딩, dog_id 필터가 WHERE 절.

프롬프트에는 지시만 — 요약 어휘의 원천은 관찰 기록. trait 축 이름(활동성 등)과
숫자 임계는 설계가 정한 스키마/계기라 허용.
"""

DB_URL_ENV = "DATABASE_URL"
DEFAULT_DB_URL = "postgresql://localhost:5432/shelter_puppy"
EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024

# trait 5축 — 설계 확정 스키마(축은 그대로, 원천만 07-08 재라우팅).
TRAITS = ("활동성", "독립성", "사교성", "안정성", "호기심")
