"""텍스트 레이아웃 — 주인공을 가리지 않는 빈 영역 선택 + 가독성 산수 (순수 결정론).

배경(2026-07-06 사용자): 저작이 8방위를 활용 못하는 원인은 프롬프트가 아니라 정보 —
텍스트 위치는 *구도*의 판단인데 저작(Gemma)은 관찰 기록만 보고 픽셀을 못 본다.
빈 곳은 YOLO(박스)+ByteTrack(트랙)+사진앵커(주인공)가 이미 찾아놨다: 주인공 박스의
여집합. 그래서 위치의 기본값은 auto 이고, 실제 자리는 이 모듈이 박스 겹침으로 정한다
("속성은 단어가 아니라 픽셀에서"의 공간판, _presence_spans 의 시간축 ↔ 여기는 공간축).

가독성(사용자 확정): 예산은 생성 앞단(저작 프롬프트)이 유도하고, 여기의 검증은
경고 위주 안전망 — LLM 은 글자 수를 못 세므로 앞단 예산은 진짜 한계의 절반쯤(마진이
부정확성을 흡수), 뒷단은 발동 시 경고를 찍어 프롬프트 회귀 신호로 쓴다(_watch_echo 결).

여기 함수들은 rect 산수만 안다 — 박스 투영·텍스트 측정(PIL)은 run.py 가 만들어 넘긴다.
rect = (x1, y1, x2, y2) 출력 캔버스 px.
"""

from __future__ import annotations

# 자동 배치 후보 순서(연출 관행 — 기하 상수이지 내용 아님): 하단이 자막의 기본 자리,
# 막히면 상단 → 하단 구석 → 좌/우 → 상단 왼쪽. top-right 는 AI 배지 상주 영역이라
# 자동 후보에서 제외(명시 지정만 허용).
AUTO_ORDER = ("bottom", "top", "bottom-left", "bottom-right",
              "left", "right", "top-left")

# 영역-주인공 겹침 허용 상한(영역 면적 대비) — 이 이하면 "빈 곳"으로 본다.
OCC_MAX = 0.10

# 자막 읽기 속도(자/초)와 표시 하한(초) [잠정 — 저작 품질처럼 취향 배터리로 보정].
# 하한은 run.MIN_SHOW(플래시 컷 1.5초)와 같은 결 — 그보다 짧은 텍스트는 플래시다.
READ_CPS = 12.0
TEXT_MIN_SHOW = 1.5


def rect_overlap(a: tuple, b: tuple) -> float:
    """두 rect 교집합 넓이(px²). 안 겹치면 0."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def occupancy(rect: tuple, boxes: list[tuple]) -> float:
    """영역이 주인공에게 가려지는 최악 순간 — max(프레임 박스와의 겹침/영역 넓이).

    boxes = 표시창 동안의 프레임별 주인공 박스(출력 좌표). 평균이 아니라 max 인
    이유: "한 순간이라도 가리면 가린 것"(플래시 컷과 같은 지각 기준). 박스가 없으면
    0 — 검출 희소 footage 는 판단 포기가 아니라 어느 영역이든 허용(안전한 저하,
    presence 필터의 "박스 없으면 생략"과 동일).
    """
    area = (rect[2] - rect[0]) * (rect[3] - rect[1])
    if area <= 0 or not boxes:
        return 0.0
    return max(rect_overlap(rect, b) for b in boxes) / area


def pick_region(rects: dict[str, tuple], boxes: list[tuple],
                taken: list = (), order: tuple = AUTO_ORDER,
                thresh: float = OCC_MAX) -> str | None:
    """빈 영역 선택 — 후보 순서대로 ①같은 시간에 보이는 텍스트(taken, 영역 이름들)와
    같거나 인접(ADJACENT)하지 않고 ②주인공 겹침 ≤ thresh 인 첫 영역. 겹침만 남으면
    최소 겹침 영역, 인접까지 전부 막히면 None(호출부가 드롭/폴백 — 붙은 자리에
    구겨 넣는 건 이 함수의 권한이 아니다).
    """
    cands = [p for p in order if p in rects
             and not any(conflicts(p, t) for t in taken)]
    for pos in cands:
        if occupancy(rects[pos], boxes) <= thresh:
            return pos
    if cands:
        return min(cands, key=lambda p: occupancy(rects[p], boxes))
    return None


def required_secs(text: str, cps: float = READ_CPS,
                  floor: float = TEXT_MIN_SHOW) -> float:
    """이 텍스트를 읽는 데 필요한 표시 시간(초) = max(하한, 글자 수/읽기 속도)."""
    return max(floor, len((text or "").strip()) / cps)


# 카피 체류 배수 — 표시 시간 = 읽기 시간 × 이 배수. 카피는 상시가 아니라 "보였다
# 사라지는" 것(2026-07-06 사용자: '심쿵 주의보!'가 장면을 넘어 계속 떠 있으니 이상,
# 뜬 순간은 딱 좋았음 — 상시는 title 의 몫). [잠정 — 취향 배터리로 보정]
DWELL_FACTOR = 2.0

# 동시 표시 상한(title 포함) — "한번에 세 개 이상은 안 되겠다"(2026-07-06 사용자 확정).
MAX_CONCURRENT = 2

# title 체류 배수 — title 도 상시가 아니라 소멸(2026-07-06 사용자 확정). 카피(2.0)보다
# 눈에 띄게 길게: title 은 문패라 위계가 높고, 인트로에선 자막을 먼저 읽고 title 로
# 눈이 온다. 짧은 title 은 하한 1.5s 에 걸려 배수가 곧 체류 시간(3.0 = 4.5초 ≈ 쇼츠
# 훅 구간). 2.5 vs 3.0 은 취향 축 — 이 상수 하나로 보정. [잠정]
TITLE_DWELL_FACTOR = 3.0


def title_window(title: str, total: float) -> tuple[float, float]:
    """title 표시창 — 시작부터 읽기 시간 × TITLE_DWELL_FACTOR (영상보다 길면 전체).

    렌더(overlay enable)와 동시성 장부(placed)가 같은 창을 봐야 하므로 단일 출처.
    title 이 떠난 뒤에는 동시 슬롯(MAX_CONCURRENT)과 top 인접 영역이 풀린다.
    """
    return (0.0, min(total, required_secs(title) * TITLE_DWELL_FACTOR))

# 인접 영역 — 4방위와 그 이웃 구석을 *같은 시간에* 함께 쓰면 번인이 겹치는 느낌
# (2026-07-06 사용자: 상단+좌·우상단, 하단+좌·우하단, 좌단+좌상·좌하단, 우단+우상·우하단).
# 시간이 겹치는 텍스트끼리만 적용 — 순차 표시는 같은 영역도 재사용 가능.
ADJACENT = {
    "top": {"top-left", "top-right"},
    "bottom": {"bottom-left", "bottom-right"},
    "left": {"top-left", "bottom-left"},
    "right": {"top-right", "bottom-right"},
    "top-left": {"top", "left"},
    "top-right": {"top", "right"},
    "bottom-left": {"bottom", "left"},
    "bottom-right": {"bottom", "right"},
}


def conflicts(a: str, b: str) -> bool:
    """두 영역을 동시에 쓸 수 없나 — 같은 자리이거나 인접."""
    return a == b or b in ADJACENT.get(a, set())


def resolve_window(w0: float, w1: float, span: list | None,
                   req: float) -> tuple[float, float] | None:
    """표시창 확정 — [w0,w1](블록 범위 실측 창)에 span 비율 적용 후 가독성 검증.

    창이 req 미만이면 span 을 버리고 범위 전체로 확장(명시 복구가 아니라 저작
    재량의 재조정이라 안전한 방향), 그래도 부족하면 None(호출부가 경고 후 드롭 —
    못 읽는 텍스트는 없느니만 못하다, "틀린 자동 < 없는 자동").
    """
    a, b = w0, w1
    if span:
        a = w0 + (w1 - w0) * span[0]
        b = w0 + (w1 - w0) * span[1]
    if b - a + 1e-6 < req:
        a, b = w0, w1
    return (a, b) if b - a + 1e-6 >= req else None


def _busy(windows: list[tuple], k: int = MAX_CONCURRENT) -> list[tuple]:
    """이미 k개 이상 보이는 구간들 — 여기에 하나 더 얹으면 상한 초과(이벤트 스윕)."""
    evs = sorted([(w[0], 1) for w in windows] + [(w[1], -1) for w in windows],
                 key=lambda e: (e[0], e[1]))          # 같은 시점은 끝(-1) 먼저
    out, cnt, start = [], 0, None
    for x, dv in evs:
        cnt += dv
        if cnt >= k and start is None:
            start = x
        elif cnt < k and start is not None:
            out.append((start, x)); start = None
    return out


def _free(rng: tuple, busy: list[tuple]) -> list[tuple]:
    """rng 에서 busy 구간들을 뺀 자유 조각들."""
    free, cur = [], rng[0]
    for b0, b1 in sorted(busy):
        if b1 <= rng[0] or b0 >= rng[1]:
            continue
        if b0 > cur:
            free.append((cur, min(b0, rng[1])))
        cur = max(cur, b1)
    if cur < rng[1]:
        free.append((cur, rng[1]))
    return free


def place_copy(w0: float, w1: float, span: list | None, text: str,
               windows: list[tuple]) -> tuple[float, float] | None:
    """카피 표시창 — 동시 상한(MAX_CONCURRENT)을 지키는 틈에 배치.

    windows = 이미 확정된 텍스트들(title·자막·선행 카피)의 표시창. 카피는 보조라
    항상 양보하는 쪽(핀 예약과 같은 결 — 아는 것의 우선권을 결정론으로 보장).
    ①span 명시 → 그 창과 자유 조각의 교집합에서 읽을 수 있는 첫 조각
    ②미지정/실패 → 가장 이른 자유 조각에서 읽기 시간 × DWELL_FACTOR 만
      "보였다 사라짐"(상시는 title 의 몫). 틈이 없으면 None(호출부 경고 드롭).
    """
    req = required_secs(text)
    free = _free((w0, w1), _busy(windows))
    if span:
        a = w0 + (w1 - w0) * span[0]
        b = w0 + (w1 - w0) * span[1]
        for p0, p1 in free:
            lo, hi = max(a, p0), min(b, p1)
            if hi - lo + 1e-6 >= req:
                return (lo, hi)
    for p0, p1 in free:
        if p1 - p0 + 1e-6 >= req:
            return (p0, p0 + min(p1 - p0, req * DWELL_FACTOR))
    return None
