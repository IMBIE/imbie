from setuptools import setup, find_packages

from imbie2.version import __version__

packages = find_packages(exclude=["*.test", "*.test.*", "test.*", "test"])

setup(
    name="imbie",
    version=__version__,
    description="IMBIE data processor",
    author="isardSAT",
    author_email="gorka.moyano@isardsat.cat",
    packages=packages,
    entry_points={
        "console_scripts": [
            "imbie = imbie2.proc.main:main",
            "imbie-preproc = imbie2.proc.pre_processed:main",
            "imbie-processdm = imbie2.proc.dm_processor:main",
        ]
    },
    install_requires=["prettytable", "numpy", "scipy", "matplotlib", "pandas"],
)
