"""Top-level conftest so pytest does not try to import the package's
``__init__.py`` as a test module (the package uses relative imports that
require it to be loaded *as a package*, not as a top-level script)."""

collect_ignore_glob = [
    "__init__.py",
    "client.py",
    "server/*",
    "data/*",
    "scripts/*",
    "notebooks/*",
    "plots/*",
    "docs/*",
]
