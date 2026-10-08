@echo off
rem 一键重建报告 PDF。
rem
rem 用法（在仓库根目录）：
rem     build.cmd
rem     build.cmd --out reports\_draft.pdf        构建到别处，方便与当前版对比
rem     build.cmd --page-background ""            关掉内页背景
rem
rem 全部参数见： .venv\Scripts\python.exe scripts\build_pdf.py --help

setlocal
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\build_pdf.py" %*
