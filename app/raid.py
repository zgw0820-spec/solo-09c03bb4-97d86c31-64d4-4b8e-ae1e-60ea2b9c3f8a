"""RAID 5 单盘故障演练引擎。

- 3 个本地文件模拟 3 块盘，每盘 4 个块（每块 4096 字节）
- 4 个条带，校验块在 3 块盘间轮换（左对称：条带0→盘2, 条带1→盘1, 条带2→盘0, 条带3→盘2）
- 每个条带 2 个数据块 + 1 个异或校验块，共 8 个逻辑块（0~7）
- 支持：摘盘后降级读写、逐条带重建替换盘、重建途中继续读写、
  注入"重建写新盘失败"、失败后保持降级并可重试
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path

BLOCK_SIZE = 4096
NUM_DISKS = 3
NUM_STRIPES = 4
BLOCKS_PER_STRIPE = NUM_DISKS - 1          # 每条带 2 个数据块
LOGICAL_BLOCKS = NUM_STRIPES * BLOCKS_PER_STRIPE  # 8 个逻辑块

ONLINE = "online"
MISSING = "missing"
REBUILDING = "rebuilding"


class RaidError(Exception):
    """演练操作被拒绝（API 层映射为 HTTP 409）。"""


class SimulatedDiskWriteError(RaidError):
    """注入的故障：重建过程中写替换盘失败。"""


def parity_disk(stripe: int) -> int:
    """条带 stripe 的校验块所在盘号（轮换：2, 1, 0, 2）。"""
    return NUM_DISKS - 1 - (stripe % NUM_DISKS)


def data_disks(stripe: int) -> list[int]:
    """条带 stripe 的两个数据盘号（升序）。"""
    return [d for d in range(NUM_DISKS) if d != parity_disk(stripe)]


def locate(logical_block: int) -> tuple[int, int]:
    """逻辑块号 -> (条带号, 所在盘号)。"""
    if not 0 <= logical_block < LOGICAL_BLOCKS:
        raise ValueError(f"逻辑块号需在 0~{LOGICAL_BLOCKS - 1} 之间，收到 {logical_block}")
    stripe = logical_block // BLOCKS_PER_STRIPE
    return stripe, data_disks(stripe)[logical_block % BLOCKS_PER_STRIPE]


def xor_bytes(a: bytes, b: bytes) -> bytes:
    return (int.from_bytes(a) ^ int.from_bytes(b)).to_bytes(BLOCK_SIZE)


class RaidArray:
    """整个演练阵列。所有状态变更与文件 IO 都在同一把锁内完成，

    保证"重建某条带"与"读写该条带"互斥，不会出现读到半新半旧的条带。
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.log_entries: deque[dict] = deque(maxlen=300)
        self.slots: list[str] = [ONLINE] * NUM_DISKS
        self.rebuilt_stripes: set[int] = set()   # 重建中的盘已补齐的条带
        self.rebuild: dict = self._fresh_rebuild_state()
        self._rebuild_thread: threading.Thread | None = None
        self._load_or_init()

    # ---------- 基础 ----------

    @staticmethod
    def _fresh_rebuild_state() -> dict:
        return {
            "active": False,      # 正在重建
            "failed": False,      # 上一次重建失败（保持降级，可重试）
            "complete": False,    # 上一次重建已圆满完成（换盘完成）
            "error": None,
            "next_stripe": 0,
            "fail_at": None,      # 注入故障的条带号
        }

    def _disk_path(self, slot: int) -> Path:
        return self.data_dir / f"disk{slot}.img"

    def _meta_path(self) -> Path:
        return self.data_dir / "meta.json"

    def _log(self, msg: str) -> None:
        self.log_entries.append({"ts": time.strftime("%H:%M:%S"), "msg": msg})

    def _save_meta(self) -> None:
        meta = {
            "slots": self.slots,
            "rebuilt_stripes": sorted(self.rebuilt_stripes),
            "rebuild": self.rebuild,
        }
        tmp = self._meta_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        tmp.replace(self._meta_path())

    def _load_or_init(self) -> None:
        meta_path = self._meta_path()
        if not meta_path.exists():
            self.initialize()
            return
        meta = json.loads(meta_path.read_text())
        self.slots = meta["slots"]
        self.rebuilt_stripes = set(meta["rebuilt_stripes"])
        self.rebuild = {**self._fresh_rebuild_state(), **meta["rebuild"]}
        for i in range(NUM_DISKS):
            exists = self._disk_path(i).exists()
            if self.slots[i] == ONLINE and not exists:
                self.slots[i] = MISSING
                self._log(f"盘 {i} 的镜像文件丢失，按缺失处理")
            elif self.slots[i] == MISSING and exists:
                self._disk_path(i).unlink()  # 已摘除盘的残留文件
            elif self.slots[i] == REBUILDING:
                # 重启打断了重建：替换盘内容不可信，丢弃并保持降级
                if exists:
                    self._disk_path(i).unlink()
                self.slots[i] = MISSING
                self.rebuilt_stripes = set()
                self.rebuild.update(active=False, failed=True,
                                    error="重建被服务重启打断，保持降级，可重试重建")
                self._log(f"盘 {i} 的重建被重启打断，已回退到降级状态")
        self.rebuild["active"] = False
        self._save_meta()

    # ---------- 物理块读写（调用方须保证槽位可用） ----------

    def _read_physical(self, slot: int, stripe: int) -> bytes:
        path = self._disk_path(slot)
        if not path.exists():
            raise RaidError(f"盘 {slot} 不在位，无法读取")
        with open(path, "rb") as f:
            f.seek(stripe * BLOCK_SIZE)
            data = f.read(BLOCK_SIZE)
        return data.ljust(BLOCK_SIZE, b"\x00")

    def _write_physical(self, slot: int, stripe: int, data: bytes) -> None:
        path = self._disk_path(slot)
        if not path.exists():
            raise RaidError(f"盘 {slot} 不在位，无法写入")
        with open(path, "r+b") as f:
            f.seek(stripe * BLOCK_SIZE)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())  # 已确认的写入要落盘

    def _available(self, slot: int, stripe: int) -> bool:
        """槽位对某条带是否可用（重建中的盘只对已重建条带可用）。"""
        return self.slots[slot] == ONLINE or (
            self.slots[slot] == REBUILDING and stripe in self.rebuilt_stripes
        )

    # ---------- 阵列初始化 ----------

    def initialize(self) -> None:
        """重置整个阵列：3 块新盘 + 演示数据（演练起点）。"""
        with self.lock:
            self.rebuild["active"] = False  # 让自动重建线程退出
            for i in range(NUM_DISKS):
                with open(self._disk_path(i), "wb") as f:
                    f.write(bytes(BLOCK_SIZE * NUM_STRIPES))
                self.slots[i] = ONLINE
            self.rebuilt_stripes = set()
            self.rebuild = self._fresh_rebuild_state()
            self._log("阵列已初始化：3 盘在线，4 条带，轮换异或校验")
            for lb in range(LOGICAL_BLOCKS):
                text = f"逻辑块{lb}演示数据|" + f"raid5-demo-block{lb}|" * 20
                self._write_logical(lb, text.encode("utf-8"))
            self._log("已写入 8 个逻辑块的演示数据")
            self._save_meta()

    # ---------- 逻辑块读写 ----------

    def read_logical(self, logical_block: int) -> tuple[bytes, str]:
        """读逻辑块，返回 (数据, 来源说明)。缺盘时用同条带异或重构。"""
        with self.lock:
            stripe, t = locate(logical_block)
            p = parity_disk(stripe)
            o = next(d for d in data_disks(stripe) if d != t)
            if self.slots[t] == ONLINE:
                return self._read_physical(t, stripe), f"盘{t}直接读取"
            # t 缺失（或在重建中）：数据 = 校验块 ⊕ 另一数据块
            if not self._available(p, stripe) or not self._available(o, stripe):
                raise RaidError(f"条带 {stripe} 冗余不足，逻辑块 {logical_block} 无法重构")
            data = xor_bytes(self._read_physical(p, stripe), self._read_physical(o, stripe))
            return data, f"盘{t}缺失，由盘{p}⊕盘{o}异或重构"

    def write_logical(self, logical_block: int, data: bytes) -> str:
        with self.lock:
            return self._write_logical(logical_block, data)

    def _write_logical(self, logical_block: int, data: bytes) -> str:
        if len(data) > BLOCK_SIZE:
            raise RaidError(f"数据超过块大小 {BLOCK_SIZE} 字节")
        data = data.ljust(BLOCK_SIZE, b"\x00")
        stripe, t = locate(logical_block)
        p = parity_disk(stripe)
        o = next(d for d in data_disks(stripe) if d != t)
        t_ok = self._available(t, stripe)
        p_ok = self._available(p, stripe)
        if t_ok and p_ok:
            # 增量更新校验：P' = P ⊕ 旧数据 ⊕ 新数据
            old_t = self._read_physical(t, stripe)
            old_p = self._read_physical(p, stripe)
            new_p = xor_bytes(old_p, xor_bytes(old_t, data))
            self._write_physical(t, stripe, data)
            self._write_physical(p, stripe, new_p)
            detail = f"写入盘{t}并更新盘{p}校验"
        elif t_ok:
            # 校验盘缺席（被摘或该条带尚未重建）：只写数据块，
            # 校验之后由重建按 P = 数据⊕数据 重新计算
            self._write_physical(t, stripe, data)
            detail = f"校验盘缺席，仅写入盘{t}"
        elif p_ok:
            # 目标数据盘缺席：把新数据折进校验 P' = 新数据 ⊕ 另一数据块，
            # 之后读取/重建可用 P'⊕另一数据块 还原新数据
            if not self._available(o, stripe):
                raise RaidError(f"条带 {stripe} 冗余不足，无法写入")
            new_p = xor_bytes(data, self._read_physical(o, stripe))
            self._write_physical(p, stripe, new_p)
            detail = f"盘{t}缺席，新数据已折入盘{p}校验"
        else:
            raise RaidError(f"条带 {stripe} 冗余不足，无法写入")
        self._log(f"写逻辑块 {logical_block}（条带{stripe}）：{detail}")
        return detail

    # ---------- 摘盘 ----------

    def remove_disk(self, slot: int) -> None:
        with self.lock:
            if not 0 <= slot < NUM_DISKS:
                raise RaidError(f"盘号需在 0~{NUM_DISKS - 1} 之间")
            if any(s != ONLINE for s in self.slots):
                raise RaidError("仅允许在阵列健康时摘盘；当前已降级或正在重建，请先完成重建")
            self._disk_path(slot).unlink(missing_ok=True)
            self.slots[slot] = MISSING
            self.rebuild["complete"] = False
            self._log(f"盘 {slot} 已摘除，阵列进入降级运行（仍可读写）")
            self._save_meta()

    # ---------- 重建 ----------

    def start_rebuild(self, fail_at: int | None = None,
                      auto: bool = False, delay: float = 0.8) -> None:
        """换上替换盘并开始逐条带重建。fail_at 用于注入某一步写新盘失败。"""
        with self.lock:
            if self.rebuild["active"]:
                raise RaidError("重建已在进行中")
            missing = [i for i, s in enumerate(self.slots) if s == MISSING]
            if len(missing) != 1:
                raise RaidError("需要且仅需要一块盘处于缺失状态才能开始重建")
            if fail_at is not None and not 0 <= fail_at < NUM_STRIPES:
                raise RaidError(f"fail_at 需在 0~{NUM_STRIPES - 1} 之间")
            m = missing[0]
            # 替换盘：全新的空盘
            with open(self._disk_path(m), "wb") as f:
                f.write(bytes(BLOCK_SIZE * NUM_STRIPES))
            self.slots[m] = REBUILDING
            self.rebuilt_stripes = set()
            self.rebuild.update(active=True, failed=False, complete=False,
                                error=None, next_stripe=0, fail_at=fail_at)
            inject = f"，注入条带 {fail_at} 写盘失败" if fail_at is not None else ""
            self._log(f"盘 {m} 已换上替换盘，开始逐条带重建{inject}")
            self._save_meta()
            if auto:
                self._rebuild_thread = threading.Thread(
                    target=self._auto_rebuild, args=(delay,), daemon=True)
                self._rebuild_thread.start()

    def _auto_rebuild(self, delay: float) -> None:
        while True:
            time.sleep(delay)
            with self.lock:
                if not self.rebuild["active"]:
                    return
                try:
                    if self._step_locked():
                        return
                except SimulatedDiskWriteError:
                    return

    def rebuild_step(self) -> None:
        """手动单步重建一个条带。失败不抛异常，状态体现在 rebuild 字段里。"""
        with self.lock:
            if not self.rebuild["active"]:
                raise RaidError("当前没有进行中的重建")
            try:
                self._step_locked()
            except SimulatedDiskWriteError:
                pass

    def _step_locked(self) -> bool:
        """重建一个条带，返回是否全部完成。注入失败时抛 SimulatedDiskWriteError。"""
        if REBUILDING not in self.slots:
            raise RaidError("没有正在重建的盘")
        m = self.slots.index(REBUILDING)
        s = self.rebuild["next_stripe"]
        if self.rebuild["fail_at"] == s:
            self._abort_rebuild_locked(m, f"模拟故障：重建条带 {s} 时写替换盘（盘 {m}）失败")
            raise SimulatedDiskWriteError(self.rebuild["error"])
        others = [d for d in range(NUM_DISKS) if d != m]
        block = xor_bytes(self._read_physical(others[0], s),
                          self._read_physical(others[1], s))
        self._write_physical(m, s, block)
        self.rebuilt_stripes.add(s)
        self.rebuild["next_stripe"] = s + 1
        self._log(f"条带 {s} 已重建到盘 {m}（{self.rebuild['next_stripe']}/{NUM_STRIPES}）")
        if self.rebuild["next_stripe"] >= NUM_STRIPES:
            self.slots[m] = ONLINE
            self.rebuilt_stripes = set()
            self.rebuild.update(active=False, failed=False, complete=True, error=None)
            self._log(f"盘 {m} 全部 {NUM_STRIPES} 个条带补齐最新内容，换盘完成，可再摘另一盘核对")
            self._save_meta()
            return True
        self._save_meta()
        return False

    def _abort_rebuild_locked(self, m: int, reason: str) -> None:
        """重建失败：丢弃不可信的替换盘，保持降级读写，允许重试。"""
        self._disk_path(m).unlink(missing_ok=True)
        self.slots[m] = MISSING
        self.rebuilt_stripes = set()
        self.rebuild.update(active=False, failed=True, complete=False,
                            error=reason, next_stripe=0)
        self._log(f"{reason}；替换盘已作废，阵列保持降级读写，可重试重建")
        self._save_meta()

    # ---------- 状态 ----------

    def status(self) -> dict:
        with self.lock:
            if REBUILDING in self.slots:
                array_state = "rebuilding"
            elif self.rebuild["failed"]:
                array_state = "rebuild_failed"
            elif MISSING in self.slots:
                array_state = "degraded"
            else:
                array_state = "healthy"
            slots = []
            for i in range(NUM_DISKS):
                path = self._disk_path(i)
                slots.append({
                    "slot": i,
                    "state": self.slots[i],
                    "file_exists": path.exists(),
                    "size": path.stat().st_size if path.exists() else None,
                    "rebuilt_stripes": sorted(self.rebuilt_stripes) if self.slots[i] == REBUILDING else [],
                })
            stripes = []
            for s in range(NUM_STRIPES):
                p = parity_disk(s)
                cells = []
                for d in range(NUM_DISKS):
                    if d == p:
                        cells.append({"disk": d, "role": "parity", "logical_block": None})
                    else:
                        k = data_disks(s).index(d)
                        cells.append({"disk": d, "role": "data",
                                      "logical_block": s * BLOCKS_PER_STRIPE + k})
                stripes.append({"stripe": s, "parity_disk": p, "cells": cells})
            return {
                "array_state": array_state,
                "block_size": BLOCK_SIZE,
                "logical_blocks": LOGICAL_BLOCKS,
                "slots": slots,
                "stripes": stripes,
                "rebuild": {**self.rebuild,
                            "rebuilt_stripes": sorted(self.rebuilt_stripes)},
                "log": list(self.log_entries)[-60:],
            }
