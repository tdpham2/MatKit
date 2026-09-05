"""Optional MCP tests use the real SDK and stdio, with synthetic engines."""

import asyncio
import base64
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip("mcp")
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from matkit.api import inspect_run
from matkit.mcp import create_server


def profile(*args):
    return {
        "executables": {
            "zeopp": [
                sys.executable,
                str(Path(__file__).parent / "fixtures" / "fake_engine.py"),
                "zeopp",
                *args,
            ]
        }
    }


def test_stdio_discovery_execution_and_artifact_retrieval(sample_cif, tmp_path):
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"default": profile()}))
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "matkit.mcp",
            "--run-root",
            str(tmp_path / "runs"),
            "--input-root",
            str(Path(sample_cif).parent),
            "--profiles",
            str(profiles),
        ],
    )

    async def check():
        async with Client(params) as client:
            catalog = await client.list_tools()
            names = {tool.name for tool in catalog.tools}
            assert {
                "matkit_pores",
                "matkit_evaluate",
                "matkit_prepare_adsorption",
            } <= names
            pores = next(
                tool for tool in catalog.tools if tool.name == "matkit_pores"
            )
            assert pores.input_schema
            response = await client.call_tool(
                "matkit_pores",
                {
                    "request": {
                        "structure": {"path": sample_cif},
                        "analyses": ["res", "sa"],
                    }
                },
            )
            assert not response.is_error, response
            data = response.structured_content
            assert data["accepted"]
            assert set(data["summary"]["results"]) == {"res", "sa"}
            ref = next(
                a for a in data["artifacts"] if a["path"] == "result.json"
            )
            resource = await client.read_resource(ref["uri"])
            content = resource.contents[0]
            encoded = (
                content.text
                if hasattr(content, "text")
                else base64.b64decode(content.blob)
            )
            result = json.loads(encoded)
            assert result["run_id"] == data["run_id"]
            assert result["payload"]["results"] == data["summary"]["results"]

    asyncio.run(check())


def test_mcp_timeout_and_path_roots(sample_cif, tmp_path):
    server = create_server(
        run_root=tmp_path / "runs",
        input_roots=[Path(sample_cif).parent],
        profiles={"default": profile("--sleep")},
        timeout_s=2,
    )

    async def check():
        async with Client(server) as client:
            bad = await client.call_tool(
                "matkit_pores",
                {"request": {"structure": {"path": "/etc/passwd"}}},
            )
            assert bad.is_error
            result = await client.call_tool(
                "matkit_pores", {"request": {"structure": {"path": sample_cif}}}
            )
            assert result.structured_content["state"] == "interrupted"
            assert not result.structured_content["accepted"]

    asyncio.run(check())


def test_mcp_selected_catalog(sample_cif, tmp_path):
    server = create_server(
        run_root=tmp_path / "runs",
        input_roots=[Path(sample_cif).parent],
        tools=["matkit_capabilities"],
    )

    async def check():
        async with Client(server) as client:
            result = await client.list_tools()
            assert [t.name for t in result.tools] == ["matkit_capabilities"]

    asyncio.run(check())


def test_mcp_cancellation_preserves_interrupted_run(sample_cif, tmp_path):
    root = tmp_path / "runs"
    server = create_server(
        run_root=root,
        input_roots=[Path(sample_cif).parent],
        profiles={"default": profile("--sleep")},
    )

    async def check():
        async with Client(server) as client:
            task = asyncio.create_task(
                client.call_tool(
                    "matkit_pores",
                    {"request": {"structure": {"path": sample_cif}}},
                )
            )
            for _ in range(200):
                if list(root.glob("*/work/started")):
                    break
                await asyncio.sleep(0.025)
            else:
                pytest.fail("worker did not start")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            for _ in range(200):
                runs = list(root.glob("*/result.json"))
                if runs:
                    assert inspect_run(runs[0].parent).state == "interrupted"
                    break
                await asyncio.sleep(0.025)
            else:
                pytest.fail("cancellation did not persist the interrupted run")

    asyncio.run(check())
