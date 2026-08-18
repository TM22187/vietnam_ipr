# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = Path(SPECPATH)
rapidocr_data = collect_data_files("rapidocr")
rapidocr_hidden = collect_submodules("rapidocr")

a = Analysis(
    [str(root / "desktop_app.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[
        (str(root / "models" / "best_vietnam_lpr.onnx"), "models"),
        (str(root / "assets" / "app.ico"), "assets"),
        *rapidocr_data,
    ],
    hiddenimports=rapidocr_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch", "torchvision", "ultralytics", "paddle", "paddleocr",
        "matplotlib", "PyQt5", "PyQt6", "PySide2", "PySide6",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VietnamLPR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    icon=str(root / "assets" / "app.ico"),
    version=str(root / "version_info.txt"),
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="VietnamLPR",
)
