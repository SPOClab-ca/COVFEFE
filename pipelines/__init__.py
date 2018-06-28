from utils.function_registry import makeRegistrar

pipeline_registry = makeRegistrar()

import importlib
import os
_pkg_name = os.path.basename(os.path.dirname(__file__))
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    importlib.import_module(_pkg_name + "." + module[:-3])
del module
