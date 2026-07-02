"""한국어 TTS 정규화기 — 숫자·단위·날짜를 한글 읽기로 (결정론, 엔진 앞단).

스파이크 실증(2026-07-02): 생 숫자("7.2킬로그램", "2026년 1월 22일")는 세 엔진 전부
비틀거렸고, 한글 읽기("칠 점 이 킬로그램")로는 전부 완벽 → 엔진 무관 필수 전처리.
"리터럴 값은 결정론 파싱" 원칙의 연장 — 숫자 읽기를 LLM/TTS 에 맡기지 않는다.

읽기 규칙:
  - 날짜(년/월/일)·소수·일반 단위 = 한자어 수사 (이천이십육 년 / 칠 점 이)
  - 세는 단위(살·마리·개·번·명·시) = 고유어 수사 (세 살, 두 마리) — 관형형(한/두/세/네…)
  - 단위 약어(kg·km·cm·%)는 먼저 한글 단위로 치환 후 수사 변환
"""

from __future__ import annotations

import re

_SINO_DIGIT = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
_SINO_SMALL = ["", "십", "백", "천"]
_SINO_BIG = ["", "만", "억", "조"]

# 고유어 수사 — 관형형(단위 앞에서 쓰는 형태). 20 이상 십 단위 + 1~9 조합.
_NATIVE_ONES = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
_NATIVE_TENS = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]

# 고유어 수사로 읽는 세는 단위 (그 외 단위는 한자어). 조사("마리와")는 허용하되,
# 한자어로 읽는 복합 단위(개월·개국·번지·번호)는 lookahead 로 제외. 시간은 시보다 먼저.
_NATIVE_COUNTERS = "시간|살|마리|개(?![월국])|번(?![지호])|명|사람|시(?!간)|가지|군데|송이"

# 단위 약어 → 한글 (수사 변환 전에 치환). 앞 공백은 소수 읽기("칠 점 이")와의
# 이음새용 — 마지막에 이중 공백을 한 칸으로 접는다.
_UNIT_ABBREV = [
    (r"(?<=[\d.])\s*kg", " 킬로그램"), (r"(?<=[\d.])\s*km", " 킬로미터"),
    (r"(?<=[\d.])\s*cm", " 센티미터"), (r"(?<=[\d.])\s*mm", " 밀리미터"),
    (r"(?<=[\d.])\s*g(?![a-z])", " 그램"), (r"(?<=[\d.])\s*%", " 퍼센트"),
]


def sino(n: int) -> str:
    """한자어 수사. 0~9999조. 1x는 '일십' 아닌 '십'(단 만 단위 첫머리 '일만'은 '만')."""
    if n == 0:
        return "영"
    parts = []
    big = 0
    while n > 0:
        chunk = n % 10000
        if chunk:
            s = ""
            for pos in range(3, -1, -1):
                d = (chunk // 10 ** pos) % 10
                if d == 0:
                    continue
                # 십·백·천 앞의 1은 생략 ("일십"→"십")
                s += ("" if d == 1 and pos > 0 else _SINO_DIGIT[d]) + _SINO_SMALL[pos]
            parts.append(s + _SINO_BIG[big])
        n //= 10000
        big += 1
    return "".join(reversed(parts))


def native(n: int) -> str:
    """고유어 수사 관형형(1~99). 범위 밖은 한자어로 폴백 — 백 마리를 '온 마리'라곤 안 읽는다."""
    if not 1 <= n <= 99:
        return sino(n)
    tens, ones = divmod(n, 10)
    return _NATIVE_TENS[tens] + _NATIVE_ONES[ones]


def _read_decimal(m: re.Match) -> str:
    whole, frac = m.group(1), m.group(2)
    return sino(int(whole)) + " 점 " + " ".join(_SINO_DIGIT[int(d)] if d != "0" else "영" for d in frac)


def normalize(text: str) -> str:
    """TTS 입력 정규화. 원문은 자막용으로 따로 보존하고, 이 결과는 합성에만 쓴다."""
    out = text
    for pat, rep in _UNIT_ABBREV:
        out = re.sub(pat, rep, out)
    # 소수 (7.2 → 칠 점 이) — 날짜보다 먼저 (마침표 혼동 방지: \d.\d 만 매칭)
    out = re.sub(r"(\d+)\.(\d+)", _read_decimal, out)
    # 고유어 세는 단위 (3살 → 세 살, 2마리와 → 두 마리와)
    out = re.sub(
        rf"(\d+)\s*({_NATIVE_COUNTERS})",
        lambda m: native(int(m.group(1))) + " " + m.group(2), out)
    # 남은 정수 전부 한자어 (2026년 → 이천이십육년, 단독 숫자 포함)
    out = re.sub(r"\d+", lambda m: sino(int(m.group(0))), out)
    return re.sub(r" {2,}", " ", out)
