import pathlib
import shutil
from pathlib import Path

import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.tmpdir import TempPathFactory
from loguru import logger

@pytest.fixture(scope="session")
def shared_cache(tmp_path_factory: TempPathFactory) -> object:
    cache_dir = tmp_path_factory.mktemp("cache", numbered=False)
    logger.info(f"创建共享缓存目录：{cache_dir}")
    yield cache_dir
    shutil.rmtree(cache_dir)
    logger.info(f"删除共享缓存目录：{cache_dir}")


@pytest.fixture(scope="session")
def resource_dir() -> pathlib.Path:
    return Path(__file__).parent / "res"


@pytest.fixture(scope="session")
def dynrender_instance(shared_cache: pathlib.Path):
    from dynrender_skia.Core import DynRender
    return DynRender(static_path=str(shared_cache))


@pytest.fixture(scope="module")
def mock_img_url() -> str:
    return "http://bilibili.com"


@pytest.fixture(scope="module")
def img_path(resource_dir: pathlib.Path) -> pathlib.Path:
    return Path(resource_dir / "bilibili.png")


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)


# @pytest.fixture
# def reportlog(pytestconfig):
#     logging_plugin = pytestconfig.pluginmanager.getplugin("logging-plugin")
#     handler_id = logger.add(logging_plugin.report_handler, format="{message}")
#     yield
#     logger.remove(handler_id)
