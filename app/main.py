"""RAID 5 单盘故障演练 — FastAPI 接口层。"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .raid import BLOCK_SIZE, LOGICAL_BLOCKS, RaidArray, RaidError, locate

STATIC_DIR = Path(__file__).parent / "static"


class RebuildRequest(BaseModel):
    fail_at: int | None = None   # 注入故障：重建到该条带时写新盘失败
    auto: bool = False           # True=自动逐条带重建；False=手动单步
    delay: float = 0.8           # 自动模式每条带间隔秒数


class WriteRequest(BaseModel):
    text: str | None = None      # 文本方式写入（UTF-8，不足一块补零）
    hex: str | None = None       # 或十六进制方式写入


def _block_view(array: RaidArray, lb: int) -> dict:
    data, source = array.read_logical(lb)
    stripe, disk = locate(lb)
    return {
        "logical_block": lb,
        "stripe": stripe,
        "disk": disk,
        "source": source,
        "size": len(data),
        "hex": data.hex(),
        "text_preview": data.decode("utf-8", errors="replace").rstrip("\x00")[:200],
    }


def create_app(data_dir: str | None = None) -> FastAPI:
    array = RaidArray(data_dir or os.environ.get("DATA_DIR", "./data"))
    app = FastAPI(title="RAID 5 单盘故障演练")

    @app.exception_handler(RaidError)
    async def raid_error_handler(_: Request, exc: RaidError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    async def status():
        return array.status()

    @app.post("/api/reset")
    async def reset():
        array.initialize()
        return array.status()

    @app.post("/api/disks/{slot}/remove")
    async def remove_disk(slot: int):
        array.remove_disk(slot)
        return array.status()

    @app.post("/api/rebuild/start")
    async def rebuild_start(req: RebuildRequest):
        array.start_rebuild(fail_at=req.fail_at, auto=req.auto, delay=req.delay)
        return array.status()

    @app.post("/api/rebuild/step")
    async def rebuild_step():
        array.rebuild_step()
        return array.status()

    @app.get("/api/blocks")
    async def read_all_blocks():
        return {"blocks": [_block_view(array, lb) for lb in range(LOGICAL_BLOCKS)]}

    @app.get("/api/blocks/{lb}")
    async def read_block(lb: int):
        return _block_view(array, lb)

    @app.put("/api/blocks/{lb}")
    async def write_block(lb: int, req: WriteRequest):
        if (req.text is None) == (req.hex is None):
            raise HTTPException(status_code=400, detail="text 与 hex 必须且只能提供一个")
        try:
            data = req.text.encode("utf-8") if req.text is not None else bytes.fromhex(req.hex)
        except ValueError:
            raise HTTPException(status_code=400, detail="hex 格式非法")
        if len(data) > BLOCK_SIZE:
            raise HTTPException(status_code=400, detail=f"数据超过块大小 {BLOCK_SIZE} 字节")
        detail = array.write_logical(lb, data)
        return {"ok": True, "detail": detail, "logical_block": lb}

    return app


app = create_app()
