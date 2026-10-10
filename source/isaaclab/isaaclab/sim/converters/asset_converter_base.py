# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import contextlib
import hashlib
import json
import logging
import os
import pathlib
import random
import re
import tempfile
import uuid
from datetime import datetime

from ...utils import to_dict, validate
from ...utils.assets import check_file_path
from ...utils.io import dump_yaml
from .asset_converter_base_cfg import AssetConverterBaseCfg

logger = logging.getLogger(__name__)


class AssetConverterBase(abc.ABC):
    """Base class for converting an asset file from different formats into USD format.

    This class provides a common interface for converting an asset file into USD. It does not
    provide any implementation for the conversion. The derived classes must implement the
    :meth:`_convert_asset` method to provide the actual conversion.

    The file conversion is lazy if the output directory (:obj:`AssetConverterBaseCfg.usd_dir`) is provided.
    The output directory records its conversions, and a lazy conversion reuses an earlier output of the same asset
    file, configuration parameters and requested USD file name, as long as the output exists and no later
    conversion wrote to its folder. Otherwise, the asset is converted again. The URDF and MJCF importers do not
    overwrite earlier outputs and write a new configuration to a new numbered folder, so each of their outputs stays
    reusable.

    To override this behavior to force conversion, the flag :obj:`AssetConverterBaseCfg.force_usd_conversion`
    can be set to True. Later lazy conversions with the same configuration reuse the forced output.

    When no output directory is defined, lazy conversion is deactivated and the generated USD file is
    stored in folder ``<tempdir>/IsaacLab/usd_{date}_{time}_{random}``, where ``<tempdir>`` is the system
    temporary directory (e.g. ``/tmp`` on POSIX, ``%TEMP%`` on Windows) and the parameters in braces are
    generated at runtime. The random identifiers help avoid a race condition where two simultaneously
    triggered conversions try to use the same directory for reading/writing the generated files.

    .. note::
        Changes to the parameters :obj:`AssetConverterBaseCfg.asset_path` and :obj:`AssetConverterBaseCfg.usd_dir`
        are not considered as modifications in the configuration instance that trigger the USD file re-generation.

    """

    def __init__(self, cfg: AssetConverterBaseCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for converting an asset file to USD format.

        Raises:
            ValueError: When provided asset file does not exist.
        """
        # check that the config is valid
        validate(cfg)
        # check if the asset file exists
        if not check_file_path(cfg.asset_path):
            raise ValueError(f"The asset path does not exist: {cfg.asset_path}")
        # save the inputs
        self.cfg = cfg

        # resolve USD directory name
        if cfg.usd_dir is None:
            # a folder in the system temp dir by the name: IsaacLab/usd_{date}_{time}_{random}
            time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._usd_dir = os.path.join(tempfile.gettempdir(), "IsaacLab", f"usd_{time_tag}_{random.randrange(10000)}")
        else:
            self._usd_dir = cfg.usd_dir

        # resolve the file name from asset file name if not provided
        if cfg.usd_file_name is None:
            usd_file_name = pathlib.PurePath(cfg.asset_path).stem
        else:
            usd_file_name = cfg.usd_file_name
        # add USD extension if not provided
        if not usd_file_name.endswith((".usd", ".usda")):
            usd_file_name += ".usd"
        self._usd_file_name = usd_file_name

        os.makedirs(self.usd_dir, exist_ok=True)
        # the record lists the conversions in the output directory, see _read_record()
        self._record_path = os.path.join(self.usd_dir, ".asset_hash")
        self._asset_hash = self._config_to_hash(cfg)
        requested_usd_file_name = pathlib.PurePath(self._usd_file_name).as_posix()

        # reuse an earlier output of the same asset file and configuration if it still exists
        cached_usd_file_name = None
        if not cfg.force_usd_conversion:
            entries, legacy_hash = self._read_record()
            cached_usd_file_name = next(
                (
                    entry["generated"]
                    for entry in entries
                    if (entry["hash"], entry["requested"]) == (self._asset_hash, requested_usd_file_name)
                    and os.path.isfile(os.path.join(self.usd_dir, entry["generated"]))
                ),
                None,
            )
            if cached_usd_file_name is None and self._can_reuse_legacy_record(legacy_hash, requested_usd_file_name):
                cached_usd_file_name = requested_usd_file_name
                # a read-only output directory keeps the record of the earlier version
                with contextlib.suppress(OSError):
                    self._write_record(requested_usd_file_name, requested_usd_file_name)

        if cached_usd_file_name is not None:
            self._usd_file_name = str(pathlib.PurePath(cached_usd_file_name))
        else:
            # convert the asset to USD
            self._convert_asset(cfg)
            # importers put the physics payloads behind a "Physics" variant set and disagree on
            # which variant to select, so settle it here
            self._select_physics_variant(cfg.physics_variant)
            # record the conversion only now: recording it earlier would let a conversion that raised
            # still count as cached, so an identical retry would skip it and return the asset
            self._write_record(requested_usd_file_name, pathlib.PurePath(self._usd_file_name).as_posix())
            # dump the configuration next to the asset, stamped with the converter that produced it
            config_path = os.path.join(self.usd_dir, "config.yaml")
            dump_yaml(config_path, to_dict(cfg))
            stamp = datetime.now().strftime("%Y-%m-%d at %H:%M:%S")
            with open(config_path, "a") as f:
                f.write(f"##\n# Generated by {self.__class__.__name__} on {stamp}.\n##\n")

    """
    Properties.
    """

    @property
    def usd_dir(self) -> str:
        """The absolute path to the directory where the generated USD files are stored."""
        return self._usd_dir

    @property
    def usd_file_name(self) -> str:
        """The file name of the generated USD file."""
        return self._usd_file_name

    @property
    def usd_path(self) -> str:
        """The absolute path to the generated USD file."""
        return os.path.join(self.usd_dir, self.usd_file_name)

    @property
    def usd_instanceable_meshes_path(self) -> str:
        """The relative path to the USD file with meshes.

        The path is with respect to the USD directory :attr:`usd_dir`. This is to ensure that the
        mesh references in the generated USD file are resolved relatively. Otherwise, it becomes
        difficult to move the USD asset to a different location.
        """
        return os.path.join(".", "Props", "instanceable_meshes.usd")

    """
    Implementation specifics.
    """

    @abc.abstractmethod
    def _convert_asset(self, cfg: AssetConverterBaseCfg):
        """Converts the asset file to USD.

        Args:
            cfg: The configuration instance for the input asset to USD conversion.
        """
        raise NotImplementedError()

    """
    Private helpers.
    """

    def _select_physics_variant(self, variant: str):
        """Author a selection for the ``"Physics"`` variant set on the converted asset.

        Importers put the physics description behind a ``"Physics"`` variant set, and which variant
        they select is not consistent: the Isaac Sim importer extensions leave the set unselected,
        which composes the asset without joints, articulation roots, or mass properties, while the
        standalone importer wheel selects one of its own. Authoring the configured variant here makes
        the outcome the same either way. The selection is authored on the asset, so a spawner that
        selects a different variant on the referencing prim still wins.

        Does nothing when the asset has no such variant set.

        Args:
            variant: The variant to select.

        Raises:
            ValueError: When the asset offers a ``"Physics"`` variant set without the requested
                variant. Substituting another one would silently hand back an asset configured for a
                different backend than the caller asked for.
        """
        from pxr import Usd

        stage = Usd.Stage.Open(self.usd_path)
        prim = stage.GetDefaultPrim()
        if not prim or "Physics" not in prim.GetVariantSets().GetNames():
            return
        variant_set = prim.GetVariantSets().GetVariantSet("Physics")
        available = variant_set.GetVariantNames()
        if variant not in available:
            raise ValueError(
                f"The converted asset has no '{variant}' physics variant. Set"
                f" {type(self.cfg).__name__}.physics_variant to one of: {available}."
            )
        if variant == variant_set.GetVariantSelection():
            return
        variant_set.SetVariantSelection(variant)
        stage.GetRootLayer().Save()

    def _read_record(self) -> tuple[list[dict[str, str]], str | None]:
        """Read the record of the conversions in the output directory.

        The record is a JSON document with one entry per conversion. An entry holds the hash of the asset file
        and configuration (see :meth:`_config_to_hash`), the requested USD file, and the USD file that the
        conversion generated, both relative to the output directory with ``/`` separators. Entries that are
        malformed or point outside the output directory are ignored. Earlier versions wrote only the hash of
        the last conversion, on the first line.

        Returns:
            The valid entries, and the hash of a record written by an earlier version or None.
        """
        try:
            with open(self._record_path, encoding="utf-8") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            return [], None
        try:
            record = json.loads(content)
        except ValueError:
            record = None
        # a hash of an earlier version can also parse as a JSON number
        if not isinstance(record, dict) or record.get("version") != 1:
            lines = content.splitlines()
            return [], lines[0].strip() if lines else None
        recorded_entries = record.get("entries")
        if not isinstance(recorded_entries, list):
            recorded_entries = []
        keys = ("hash", "requested", "generated")
        entries = []
        for entry in recorded_entries:
            if not isinstance(entry, dict) or not all(isinstance(entry.get(key), str) for key in keys):
                continue
            generated = pathlib.PurePath(entry["generated"])
            # a record edited by hand must not point outside the output directory
            if generated.anchor or ".." in generated.parts:
                continue
            entries.append({key: entry[key] for key in keys})
        return entries, None

    def _can_reuse_legacy_record(self, legacy_hash: str | None, requested_usd_file_name: str) -> bool:
        """Check whether a record of an earlier version can be reused for the requested USD file.

        Earlier versions recorded only the hash of the last conversion, not its USD file, and loaded the requested
        USD file when the hash matched. The record is not reused while a numbered folder, such as ``<name>_1`` next
        to ``<name>``, shows that a later conversion went elsewhere. As in earlier versions, the requested file may
        still hold an earlier conversion if such a folder was deleted, or if the last conversion had this hash but
        wrote another requested file.

        Args:
            legacy_hash: The hash of a record written by an earlier version, or None.
            requested_usd_file_name: The requested USD file, relative to the output directory with ``/``
                separators.

        Returns:
            True if the requested USD file exists, the record holds the hash of this asset file and configuration,
            and nothing shows that the requested file holds another conversion.
        """
        if legacy_hash is None or not os.path.isfile(os.path.join(self.usd_dir, requested_usd_file_name)):
            return False
        folder = pathlib.PurePosixPath(requested_usd_file_name).parent
        if folder.name:
            parent_dir = os.path.join(self.usd_dir, folder.parent)
            numbered_folder = re.compile(rf"{re.escape(folder.name)}_\d+")
            try:
                names = os.listdir(parent_dir)
            except OSError:
                # without a listing, a numbered folder cannot be ruled out
                return False
            for name in names:
                if numbered_folder.fullmatch(name) and os.path.isdir(os.path.join(parent_dir, name)):
                    return False
        # only a lazy conversion could have recorded a hash that a lazy conversion matches
        return legacy_hash == self._config_to_hash(self.cfg, legacy=True)

    def _write_record(self, requested_usd_file_name: str, generated_usd_file_name: str) -> None:
        """Add a conversion to the record of the output directory.

        The record is read again right before writing, to keep the entries that another process added in the
        meantime. Entries whose USD file no longer exists or that this conversion replaces are dropped, and so are
        the entries in the folder this conversion wrote to: outputs in one folder can share files, such as the
        ``Props/instanceable_meshes.usd`` that every output of the mesh converter references. The URDF and MJCF
        importers give each conversion its own folder. The record is replaced in one step, so a crash cannot leave
        it half written, but writes are not serialized: two concurrent writers can still lose an entry, or keep one
        that the other writer dropped.

        Args:
            requested_usd_file_name: The requested USD file, relative to the output directory with ``/``
                separators.
            generated_usd_file_name: The generated USD file, relative to the output directory with ``/``
                separators.
        """
        entries, _ = self._read_record()
        folder = pathlib.PurePosixPath(generated_usd_file_name).parent
        entries = [
            entry
            for entry in entries
            if os.path.isfile(os.path.join(self.usd_dir, entry["generated"]))
            and (entry["hash"], entry["requested"]) != (self._asset_hash, requested_usd_file_name)
            and pathlib.PurePosixPath(entry["generated"]).parent != folder
        ]
        entries.append(
            {"hash": self._asset_hash, "requested": requested_usd_file_name, "generated": generated_usd_file_name}
        )
        # unlike the tempfile module, open() honors the umask, so other users of the directory can read the record
        temp_path = f"{self._record_path}.{uuid.uuid4().hex}.tmp"
        try:
            with open(temp_path, "x", encoding="utf-8") as f:
                json.dump({"version": 1, "entries": entries}, f, indent=2)
            os.replace(temp_path, self._record_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(temp_path)
            raise

    @staticmethod
    def _config_to_hash(cfg: AssetConverterBaseCfg, legacy: bool = False) -> str:
        """Converts the configuration object and asset file to an MD5 hash of a string.

        The hash leaves out :attr:`AssetConverterBaseCfg.force_usd_conversion`, which decides whether a
        conversion runs, not what it produces.

        .. warning::
            It only checks the main asset file (:attr:`cfg.asset_path`).

        Args:
            cfg: The asset converter configuration object.
            legacy: Whether to compute the hash that earlier versions recorded for a lazy conversion, which
                included the flag.

        Returns:
            An MD5 hash of a string.
        """

        # convert to dict and remove path related info
        config_dic = to_dict(cfg)
        _ = config_dic.pop("asset_path")
        _ = config_dic.pop("usd_dir")
        _ = config_dic.pop("usd_file_name")
        if legacy:
            # set the flag in place, which keeps the key order and so the hash of earlier versions
            config_dic["force_usd_conversion"] = False
        else:
            _ = config_dic.pop("force_usd_conversion")
        # convert config dic to bytes
        config_bytes = json.dumps(config_dic).encode()
        # hash config
        md5 = hashlib.md5()
        md5.update(config_bytes)

        # read the asset file to observe changes
        with open(cfg.asset_path, "rb") as f:
            while True:
                # read 64kb chunks to avoid memory issues for the large files!
                data = f.read(65536)
                if not data:
                    break
                md5.update(data)
        # return the hash
        return md5.hexdigest()
