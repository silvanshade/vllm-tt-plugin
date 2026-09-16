# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

"""Spawn-safe topology discovery for vLLM standard multi-process DP."""

__all__ = (
    "format_tt_visible_devices",
    "parse_mesh_grid",
    "resolve_single_device_dp_assignments",
    "standard_dp_rank_environments",
    "run_standard_dp_visible_device_group_discovery",
    "split_standard_dp_discovery_result",
    "StandardDPAssignmentT",
)

import ast
import logging
import os
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path

logger = logging.getLogger(__name__)

StandardDPAssignmentT = tuple[str, tuple[int, int]]

_MESH_GRID_PRESETS = {
    "N150": (1, 1),
    "P100": (1, 1),
    "P150": (1, 1),
    "P150x2": (1, 2),
    "N300": (1, 2),
    "P300": (1, 2),
    "N150x4": (1, 4),
    "P150x4": (1, 4),
    "T3K": (1, 8),
    "P150x8": (1, 8),
    "P300x2": (1, 4),
}


def resolve_single_device_dp_assignments(
    device_ids: object,
    data_parallel_size: int,
    visible_devices: str | None,
) -> list[StandardDPAssignmentT]:
    """Resolve explicit physical PCIe IDs without opening a connected mesh.

    # Specification
    - provides: one 1x1 assignment per rank, in the requested physical-ID order.
    - fails: ValueError for non-integer/negative/duplicate IDs, a rank-count
      mismatch, or IDs outside an inherited visibility restriction.
    - intension: performs no device enumeration or runtime initialization.
    """
    if (
        not isinstance(device_ids, list)
        or not device_ids
        or len(device_ids) != data_parallel_size
        or any(type(device_id) is not int or device_id < 0 for device_id in device_ids)
    ):
        raise ValueError(
            "tt.dp_device_ids must contain one non-negative integer physical "
            f"PCIe ID per DP rank; got {device_ids!r} for DP={data_parallel_size}"
        )
    if len(set(device_ids)) != len(device_ids):
        raise ValueError(f"tt.dp_device_ids must not share devices: {device_ids!r}")
    if visible_devices is not None:
        try:
            allowed = {int(part.strip()) for part in visible_devices.split(",")}
        except ValueError as exc:
            raise ValueError(
                "TT_VISIBLE_DEVICES must contain physical PCIe IDs: "
                f"{visible_devices!r}"
            ) from exc
        if not set(device_ids).issubset(allowed):
            raise ValueError(
                f"tt.dp_device_ids={device_ids!r} exceeds inherited "
                f"TT_VISIBLE_DEVICES={visible_devices!r}"
            )
    return [(str(device_id), (1, 1)) for device_id in device_ids]


def standard_dp_rank_environments(
    num_ranks: int, environ: Mapping[str, str]
) -> list[dict[str, str]]:
    """Snapshot rank-local runtime namespaces before workers narrow their env.

    # Specification
    - requires: num_ranks is positive.
    - provides: distinct rank-local log roots and consecutive Inspector ports;
      port zero retains the runtime's ephemeral-port selection.
    - provides: rank subdirectories for explicitly configured TT_CACHE_PATH and
      TT_METAL_CACHE, keeping concurrent cache writers independent.
    - fails: ValueError when the Inspector host/port or rank-offset range is invalid.
    - intension: does not create directories, bind sockets or initialize devices.
    """
    rpc_address_env = "TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS"
    address = environ.get(rpc_address_env, "localhost:50051")
    host, separator, port_text = address.partition(":")
    try:
        port = int(port_text) if separator else 50051
    except ValueError as exc:
        raise ValueError(f"Invalid Inspector host[:port]: {address!r}") from exc
    if not host or port < 0 or (port and port + num_ranks - 1 > 65535):
        raise ValueError(
            f"Inspector address {address!r} cannot provide ports for "
            f"{num_ranks} DP ranks"
        )
    log_root = Path(environ.get("TT_METAL_LOGS_PATH") or os.getcwd())
    cache_roots = {
        key: Path(environ[key])
        for key in ("TT_CACHE_PATH", "TT_METAL_CACHE")
        if environ.get(key)
    }
    return [
        {
            "TT_METAL_LOGS_PATH": str(log_root / f"dp_rank_{rank}"),
            rpc_address_env: f"{host}:{port + rank if port else 0}",
            **{key: str(root / f"dp_rank_{rank}") for key, root in cache_roots.items()},
        }
        for rank in range(num_ranks)
    ]


def parse_mesh_grid(
    mesh_device_env: str | None,
    num_devices_available: int,
    *,
    tg_mesh_grid: tuple[int, int],
) -> tuple[int, int]:
    """Parses one TT mesh preset or tuple into a concrete grid.

    Examples
    --------
    >>> parse_mesh_grid("T3K", 8, tg_mesh_grid=(4, 8))
    (1, 8)
    >>> parse_mesh_grid("(2, 4)", 8, tg_mesh_grid=(4, 8))
    (2, 4)
    """
    mesh_grid_dict = dict(_MESH_GRID_PRESETS)
    mesh_grid_dict["TG"] = tg_mesh_grid
    # BH Galaxy is 32x P150 in the same 32-chip topology as the WH Galaxy, so it
    # takes the caller's Galaxy grid too.
    mesh_grid_dict["BH-Galaxy"] = tg_mesh_grid

    if mesh_device_env is None:
        return (1, num_devices_available)

    try:
        parsed_value = ast.literal_eval(mesh_device_env)
        if isinstance(parsed_value, (tuple, list)) and len(parsed_value) == 2:
            return tuple(int(dim) for dim in parsed_value)
        raise ValueError("Not a valid tuple")

    except (ValueError, SyntaxError, TypeError):
        mesh_grid = mesh_grid_dict.get(mesh_device_env)
        if mesh_grid is None:
            raise ValueError(
                f"Invalid MESH_DEVICE: {mesh_device_env}. "
                f"Expected one of: {list(mesh_grid_dict.keys())}"
            ) from None
        return mesh_grid


def _resolve_parent_mesh_grid(
    mesh_device_env: str | None,
    num_devices_available: int,
) -> tuple[int, int]:
    """Normalizes the parent mesh grid to the visible device count.

    Examples
    --------
    >>> _resolve_parent_mesh_grid("T3K", 8)
    (1, 8)
    """
    mesh_grid = parse_mesh_grid(
        mesh_device_env,
        num_devices_available,
        tg_mesh_grid=(4, 8),
    )

    if mesh_grid[0] * mesh_grid[1] != num_devices_available:
        mesh_grid = (1, num_devices_available)

    return mesh_grid


def _maybe_reorder_standard_dp_visible_device_groups(
    device_groups: list[StandardDPAssignmentT],
    mesh_grid: tuple[int, int],
    data_parallel_size: int,
) -> list[StandardDPAssignmentT]:
    """Reorders TT single-host DP groups for known hardware layouts.

    Examples
    --------
    >>> groups = [("0,1", (1, 2)), ("2,3", (1, 2))]
    >>> _maybe_reorder_standard_dp_visible_device_groups(groups, (1, 2), 2) == groups
    True
    >>> # Example for WH Galaxy DP=4, which is a known special case where the default
    >>> # row-major order is not mesh-id order.
    >>> groups = [
    ...     ("0,1,2,3,4,5,6,7", (1, 8)),
    ...     ("8,9,10,11,12,13,14,15", (1, 8)),
    ...     ("16,17,18,19,20,21,22,23", (1, 8)),
    ...     ("24,25,26,27,28,29,30,31", (1, 8)),
    ... ]
    >>> _maybe_reorder_standard_dp_visible_device_groups(groups, (4, 8), 4)
    [('0,1,2,3,4,5,6,7', (1, 8)),
     ('16,17,18,19,20,21,22,23', (1, 8)),
     ('24,25,26,27,28,29,30,31', (1, 8)),
     ('8,9,10,11,12,13,14,15', (1, 8))]
    """
    import ttnn

    if (
        ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
        and mesh_grid == (4, 8)
        and data_parallel_size == 4
        and len(device_groups) == 4
    ):
        reordered_groups = [device_groups[index] for index in (0, 2, 3, 1)]
        logger.info(
            "Reordered TT single-host DP device groups for WH Galaxy DP=4 "
            "from row-major %s to mesh-id order %s",
            [visible_devices for visible_devices, _shape in device_groups],
            [visible_devices for visible_devices, _shape in reordered_groups],
        )
        return reordered_groups

    return device_groups


def split_standard_dp_discovery_result(
    discovery_result: list[str] | list[StandardDPAssignmentT] | None,
) -> tuple[list[str] | None, dict[str, tuple[int, int]]]:
    """Splits discovery output into visible-device and mesh-grid views.

    Examples
    --------
    >>> split_standard_dp_discovery_result(None)
    (None, {})
    >>> split_standard_dp_discovery_result([("0,1", (1, 2))])
    (["0,1"], {"0,1": (1, 2)})
    """
    if discovery_result is None:
        return None, {}
    if not discovery_result:
        return [], {}

    first_entry = discovery_result[0]
    if isinstance(first_entry, str):
        return discovery_result, {}

    assignments = discovery_result
    return (
        [visible_devices for visible_devices, _mesh_grid in assignments],
        {visible_devices: mesh_grid for visible_devices, mesh_grid in assignments},
    )


def _discover_standard_dp_visible_device_groups(
    mesh_device_env: str | None,
    data_parallel_size: int,
) -> list[StandardDPAssignmentT]:
    """Discovers TT visible-device groups for one single-host DP layout.

    Notes
    -----
    This helper requires a live TT runtime and submesh creation support.

    Examples
    --------
    >>> _discover_standard_dp_visible_device_groups("T3K", 4)
    [("0,1,2,3,4,5,6,7", (1, 8)), ...]
    """
    import ttnn
    from models.tt_transformers.tt.generator import create_submeshes

    mesh_device = None
    submeshes = []

    try:
        num_devices_available = ttnn.get_num_devices()
        mesh_grid = _resolve_parent_mesh_grid(mesh_device_env, num_devices_available)
        mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_grid))
        submeshes = create_submeshes(mesh_device, data_parallel_size)
        if len(submeshes) != data_parallel_size:
            raise RuntimeError(
                "TT create_submeshes returned "
                f"{len(submeshes)} groups for data_parallel_size={data_parallel_size}"
            )

        device_groups = []
        for dp_rank, submesh in enumerate(submeshes):
            device_ids = list(submesh.get_device_ids())
            if not device_ids:
                raise RuntimeError(f"TT DP rank {dp_rank} resolved to an empty submesh")
            device_groups.append(
                (
                    format_tt_visible_devices(device_ids),
                    tuple(int(dim) for dim in submesh.shape),
                )
            )

        device_groups = _maybe_reorder_standard_dp_visible_device_groups(
            device_groups,
            mesh_grid,
            data_parallel_size,
        )

        logger.info(
            "Resolved TT single-host DP device groups: %s",
            [
                f"{visible_devices}@{mesh_shape}"
                for visible_devices, mesh_shape in device_groups
            ],
        )

        return device_groups

    finally:
        for submesh in submeshes:
            with suppress(Exception):
                ttnn.close_mesh_device(submesh)
        if mesh_device is not None:
            with suppress(Exception):
                ttnn.close_mesh_device(mesh_device)


def format_tt_visible_devices(device_ids: Sequence[int | str]) -> str:
    """Serialize device identifiers for ``TT_VISIBLE_DEVICES``."""
    return ",".join(str(device_id) for device_id in device_ids)


def run_standard_dp_visible_device_group_discovery(
    conn,
    mesh_device_env: str | None,
    data_parallel_size: int,
) -> None:
    """Sends discovered TT device groups back over a pipe.

    This is the spawned child-process entrypoint for standard-DP discovery.
    """
    try:
        conn.send(
            (
                "ok",
                _discover_standard_dp_visible_device_groups(
                    mesh_device_env, data_parallel_size
                ),
            )
        )
    except Exception as exc:
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()
