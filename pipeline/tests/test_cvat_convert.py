"""CVAT 변환기 검증 — 합성 'CVAT for video 1.1' XML 로 keyframe/label 필터 확인."""

from pipeline.harness.convert.cvat_to_jsonl import parse_cvat_video_xml


# dog track 0: frame0,15 keyframe(present), frame30 outside.
# cat track 1: frame0 keyframe(present) — 라벨 필터로 제외돼야 함.
# 보간 프레임(keyframe=0)은 무시돼야 함.
CVAT_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <track id="0" label="dog">
    <box frame="0"  keyframe="1" outside="0" xtl="10" ytl="20" xbr="30" ybr="60"/>
    <box frame="7"  keyframe="0" outside="0" xtl="12" ytl="20" xbr="32" ybr="60"/>
    <box frame="15" keyframe="1" outside="0" xtl="14" ytl="20" xbr="34" ybr="60"/>
    <box frame="30" keyframe="1" outside="1" xtl="0"  ytl="0"  xbr="0"  ybr="0"/>
  </track>
  <track id="1" label="cat">
    <box frame="0"  keyframe="1" outside="0" xtl="100" ytl="100" xbr="140" ybr="160"/>
  </track>
</annotations>
"""


def test_keyframe_and_label_filter(tmp_path):
    xml = tmp_path / "ann.xml"
    xml.write_text(CVAT_XML, encoding="utf-8")
    frames = parse_cvat_video_xml(xml, fps=30.0, labels={"dog"})

    # 라벨된 프레임 = keyframe 집합 {0,15,30} (보간 7 제외).
    idxs = [f.frame_idx for f in frames]
    assert idxs == [0, 15, 30]

    # frame0: dog 박스 1개 (cat 제외). bbox = [10,20,20,40].
    f0 = frames[0]
    assert len(f0.detections) == 1
    d = f0.detections[0]
    assert d.track_id == 0 and d.cls == "dog"
    assert d.bbox.to_list() == [10.0, 20.0, 20.0, 40.0]

    # frame30: dog outside → 빈 프레임(라벨됐으나 개 없음 = true negative).
    assert frames[2].frame_idx == 30 and frames[2].detections == []

    # t = frame_idx / fps
    assert abs(frames[1].t - 15 / 30.0) < 1e-9


# original_size 가 있는 XML — 원본(1920×1080)에 라벨한 경우.
CVAT_XML_SIZED = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta><task><original_size><width>1920</width><height>1080</height></original_size></task></meta>
  <track id="0" label="dog">
    <box frame="0" keyframe="1" outside="0" xtl="200" ytl="100" xbr="400" ybr="300"/>
  </track>
</annotations>
"""


def test_rescale_to_analysis_size(tmp_path):
    xml = tmp_path / "ann.xml"
    xml.write_text(CVAT_XML_SIZED, encoding="utf-8")
    # 1920×1080 → 768×432 (균일 0.4배). box [200,100,200,200] → [80,40,80,80].
    from pipeline.harness.convert.cvat_to_jsonl import parse_cvat_video_xml as P
    frames = P(xml, fps=30.0, labels={"dog"}, target_size=(768, 432))
    d = frames[0].detections[0]
    assert d.bbox.to_list() == [80.0, 40.0, 80.0, 80.0]


def test_no_rescale_when_target_matches(tmp_path):
    xml = tmp_path / "ann.xml"
    xml.write_text(CVAT_XML_SIZED, encoding="utf-8")
    from pipeline.harness.convert.cvat_to_jsonl import parse_cvat_video_xml as P
    frames = P(xml, fps=30.0, labels={"dog"}, target_size=(1920, 1080))
    d = frames[0].detections[0]
    assert d.bbox.to_list() == [200.0, 100.0, 200.0, 200.0]
