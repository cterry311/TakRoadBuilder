# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run_tei.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
EXCLUDE = ("cudart", "cublas", "cudnn", "cufft", "curand",
           "cusparse", "cusolver", "nvrtc", "nvjitlink")
a.binaries = [b for b in a.binaries
              if not any(x in b[0].lower() for x in EXCLUDE)]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='RoadBuilder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
