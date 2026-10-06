"""打包 / 工程化回归（#23）
=======================

守住「安装即可用」的打包承诺：

* ``forgecore`` 必须作为顶层包可被导入（它曾因漏写进 wheel.packages 而在纯 wheel
  安装后 ``import amforge.forge_adapter`` 失败——本文件用导入断言锁死该回归）；
* ``diffmech`` 顶层包可导入；
* ``amforge`` 全部内置求解器模块导入就绪（import_status 为空）；
* 命令行入口 ``amforge.cli:main`` 可调用，且 ``amforge info`` 退出码为 0。

注意：物理前向的「安装可用」由 test_e2e_chain 守护；CLI 的 ``selftest`` 子命令
在构建验证阶段手动跑通。本文件只做轻量、确定、快速的打包层断言，不触发重物理。
"""

from __future__ import annotations

import amforge as af  # noqa: E402
from amforge import cli as cli_mod  # noqa: E402


def test_forgecore_packaged_and_importable():
    """forgecore 是 amforge.forge_adapter 的硬依赖，必须随 wheel 一起发布。"""
    import forgecore  # noqa: F401
    assert hasattr(forgecore, "REGISTRY"), "forgecore.REGISTRY 缺失"


def test_diffmech_packaged_and_importable():
    import diffmech  # noqa: F401
    assert diffmech.__version__ is not None


def test_amforge_import_clean():
    """所有内置求解器模块必须就绪（无依赖缺失导致的静默降级）。"""
    assert af.import_status() == {}, f"内置模块导入失败: {af.import_status()}"


def test_cli_entry_callable():
    assert callable(cli_mod.main)


def test_cli_info_exit_zero(capsys):
    """amforge info 应正常打印版本/求解器清点并退出 0（不触发重物理）。"""
    rc = cli_mod.main(["info"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "AMForge" in out
    assert "求解器" in out
