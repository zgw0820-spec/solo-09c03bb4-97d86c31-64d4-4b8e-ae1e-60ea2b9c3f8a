"""RAID 5 演练引擎与 API 的实测。"""
import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.raid import (BLOCK_SIZE, LOGICAL_BLOCKS, NUM_STRIPES, RaidArray,
                      RaidError, data_disks, locate, parity_disk)


def pattern(lb: int, tag: str) -> bytes:
    return f"块{lb}|{tag}|".encode() * 7  # 远小于 4096，写入时补零


def padded(data: bytes) -> bytes:
    return data.ljust(BLOCK_SIZE, b"\x00")


def read_all(array: RaidArray) -> list[bytes]:
    return [array.read_logical(lb)[0] for lb in range(LOGICAL_BLOCKS)]


def write_pattern_set(array: RaidArray, tag: str) -> list[bytes]:
    expected = []
    for lb in range(LOGICAL_BLOCKS):
        data = pattern(lb, tag)
        array.write_logical(lb, data)
        expected.append(padded(data))
    return expected


@pytest.fixture
def array(tmp_path):
    a = RaidArray(tmp_path / "data")
    a.initialize()
    return a


# ---------- 布局 ----------

def test_parity_rotation_and_mapping():
    assert [parity_disk(s) for s in range(NUM_STRIPES)] == [2, 1, 0, 2]
    assert [data_disks(s) for s in range(NUM_STRIPES)] == [[0, 1], [0, 2], [1, 2], [0, 1]]
    assert [locate(lb) for lb in range(LOGICAL_BLOCKS)] == [
        (0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2), (3, 0), (3, 1)]
    with pytest.raises(ValueError):
        locate(8)


# ---------- 缺盘后的降级读写 ----------

def test_degraded_read_write(array):
    expected = write_pattern_set(array, "v1")
    array.remove_disk(1)
    assert array.status()["array_state"] == "degraded"
    # 缺盘后仍能读出全部数据，包括原本落在盘1上的块（1、4、7）
    assert read_all(array) == expected
    _, source = array.read_logical(1)
    assert "重构" in source
    # 降级中写入：覆盖落在缺失盘上的块与落在在线盘上的块
    for lb in (1, 4, 0, 5):
        data = pattern(lb, "degraded-write")
        array.write_logical(lb, data)
        expected[lb] = padded(data)
    assert read_all(array) == expected


def test_remove_rules(array):
    array.remove_disk(0)
    with pytest.raises(RaidError):  # 已降级，禁止再摘第二块
        array.remove_disk(1)
    with pytest.raises(RaidError):  # 盘号越界
        array.remove_disk(5)
    with pytest.raises(RaidError):  # 已有缺盘，不能重复开重建以外的操作
        array.remove_disk(0)


# ---------- 完整重建 + 再摘另一盘核对 ----------

def test_full_rebuild_then_remove_another_disk(array):
    expected = write_pattern_set(array, "v1")
    array.remove_disk(0)
    array.start_rebuild()
    for _ in range(NUM_STRIPES):
        array.rebuild_step()
    st = array.status()
    assert st["array_state"] == "healthy" and st["rebuild"]["complete"] is True
    # 替换盘逐条带内容与另外两盘异或一致
    raw = [open(array._disk_path(i), "rb").read() for i in range(3)]
    for s in range(NUM_STRIPES):
        seg = lambda d: raw[d][s * BLOCK_SIZE:(s + 1) * BLOCK_SIZE]
        assert int.from_bytes(seg(0)) == int.from_bytes(seg(1)) ^ int.from_bytes(seg(2))
    # 再摘另一盘，核对数据
    array.remove_disk(2)
    assert read_all(array) == expected


# ---------- 重建途中继续读写（含已重建条带） ----------

def test_writes_during_rebuild(array):
    expected = write_pattern_set(array, "v1")
    array.remove_disk(1)
    array.start_rebuild()
    array.rebuild_step()  # 只重建条带 0
    # 已重建条带上的写入：块0（落盘0）、块1（落重建中的盘1）
    for lb in (0, 1):
        data = pattern(lb, "mid-rebuilt-stripe")
        array.write_logical(lb, data)
        expected[lb] = padded(data)
    # 未重建条带上的写入：块4（落重建中的盘1）、块5（落盘2）
    for lb in (4, 5):
        data = pattern(lb, "mid-pending-stripe")
        array.write_logical(lb, data)
        expected[lb] = padded(data)
    # 重建途中读取
    assert read_all(array) == expected
    for _ in range(NUM_STRIPES - 1):
        array.rebuild_step()
    assert array.status()["rebuild"]["complete"] is True
    # 重建完成后数据仍是途中写入的最新值
    assert read_all(array) == expected
    # 再摘另一盘核对：替换盘上的内容也是最新的
    array.remove_disk(0)
    assert read_all(array) == expected


# ---------- 注入重建失败 → 保持降级 → 重试 ----------

def test_rebuild_failure_and_retry(array):
    expected = write_pattern_set(array, "v1")
    array.remove_disk(2)
    array.start_rebuild(fail_at=1)
    array.rebuild_step()   # 条带 0 成功
    array.rebuild_step()   # 条带 1 注入失败
    st = array.status()
    assert st["array_state"] == "rebuild_failed"
    assert "条带 1" in st["rebuild"]["error"]
    assert st["slots"][2]["state"] == "missing"          # 替换盘已作废
    assert not array._disk_path(2).exists()
    # 失败后保持降级读写
    assert read_all(array) == expected
    confirmed = pattern(6, "confirmed-after-failure")
    array.write_logical(6, confirmed)
    expected[6] = padded(confirmed)
    assert read_all(array) == expected
    # 重试重建（不再注入故障）
    array.start_rebuild()
    for _ in range(NUM_STRIPES):
        array.rebuild_step()
    assert array.status()["rebuild"]["complete"] is True
    # 已确认的写入仍能读回；再摘另一盘核对
    array.remove_disk(1)
    assert read_all(array) == expected


def test_rebuild_requires_single_missing_disk(array):
    with pytest.raises(RaidError):  # 没有缺盘
        array.start_rebuild()
    array.remove_disk(0)
    array.start_rebuild()
    with pytest.raises(RaidError):  # 重建中不能重复开始
        array.start_rebuild()


def test_auto_rebuild(array):
    write_pattern_set(array, "v1")
    array.remove_disk(0)
    array.start_rebuild(auto=True, delay=0.05)
    deadline = time.time() + 5
    while time.time() < deadline:
        if array.status()["rebuild"]["complete"]:
            break
        time.sleep(0.05)
    assert array.status()["rebuild"]["complete"] is True


# ---------- 重启恢复 ----------

def test_restart_during_rebuild(tmp_path):
    d = tmp_path / "data"
    a = RaidArray(d)
    a.initialize()
    expected = write_pattern_set(a, "v1")
    a.remove_disk(1)
    a.start_rebuild()
    a.rebuild_step()
    # 模拟服务重启：重建中断 → 回退降级，数据不丢
    b = RaidArray(d)
    st = b.status()
    assert st["slots"][1]["state"] == "missing"
    assert st["rebuild"]["failed"] and "重启" in st["rebuild"]["error"]
    assert read_all(b) == expected
    b.start_rebuild()
    for _ in range(NUM_STRIPES):
        b.rebuild_step()
    assert read_all(b) == expected


# ---------- API 端到端 ----------

def test_api_end_to_end(tmp_path):
    client = TestClient(create_app(str(tmp_path / "api-data")))
    assert client.get("/").status_code == 200
    st = client.get("/api/status").json()
    assert st["array_state"] == "healthy"

    # 写入并读回（text 与 hex 两种方式）
    assert client.put("/api/blocks/3", json={"text": "值班员演练数据"}).status_code == 200
    r = client.get("/api/blocks/3").json()
    assert r["text_preview"] == "值班员演练数据" and r["source"] == "盘2直接读取"
    assert client.put("/api/blocks/4", json={"hex": "deadbeef"}).status_code == 200
    assert client.get("/api/blocks/4").json()["hex"].startswith("deadbeef")
    assert client.put("/api/blocks/9", json={"text": "x"}).status_code == 400
    assert client.put("/api/blocks/3", json={}).status_code == 400

    # 摘盘 → 降级读写
    assert client.post("/api/disks/1/remove").status_code == 200
    assert client.post("/api/disks/0/remove").status_code == 409  # 降级中禁止再摘
    r = client.get("/api/blocks/4").json()  # 块4 物理在盘1上
    assert r["hex"].startswith("deadbeef") and "重构" in r["source"]
    assert client.put("/api/blocks/4", json={"text": "降级写入"}).status_code == 200

    # 注入失败 → 保持降级 → 重试成功
    assert client.post("/api/rebuild/start", json={"fail_at": 2}).status_code == 200
    client.post("/api/rebuild/step"); client.post("/api/rebuild/step")
    st = client.post("/api/rebuild/step").json()
    assert st["rebuild"]["failed"] and "条带 2" in st["rebuild"]["error"]
    assert client.get("/api/blocks/4").json()["text_preview"] == "降级写入"
    assert client.post("/api/rebuild/start", json={}).status_code == 200
    for _ in range(4):
        st = client.post("/api/rebuild/step").json()
    assert st["rebuild"]["complete"] and st["array_state"] == "healthy"

    # 再摘另一盘核对全部数据
    assert client.post("/api/disks/0/remove").status_code == 200
    blocks = client.get("/api/blocks").json()["blocks"]
    assert blocks[3]["text_preview"] == "值班员演练数据"
    assert blocks[4]["text_preview"] == "降级写入"
