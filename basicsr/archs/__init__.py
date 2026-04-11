import importlib
import warnings
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]


def _is_optional_missing_dep(exc: ModuleNotFoundError) -> bool:
    missing = getattr(exc, 'name', '') or ''
    optional_prefixes = (
        'mamba_ssm',
        'causal_conv1d',
    )
    return any(missing.startswith(prefix) for prefix in optional_prefixes)


# import all the arch modules; skip optional dependency failures so unrelated
# models (e.g., PAN/SwinIR/HAT) can still be trained without mamba packages.
_arch_modules = []
for file_name in arch_filenames:
    module_name = f'basicsr.archs.{file_name}'
    try:
        _arch_modules.append(importlib.import_module(module_name))
    except ModuleNotFoundError as exc:
        if _is_optional_missing_dep(exc):
            warnings.warn(
                f'Skip optional arch module {module_name} because dependency '
                f'"{exc.name}" is not installed.',
                RuntimeWarning,
            )
            continue
        raise


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
