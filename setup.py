from setuptools import setup, find_packages

setup(
    name='imbie',
    version='0.1.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'imbie = imbie.__main__:main'
        ]
    },
    install_requires=['numpy', 'scipy', 'matplotlib']
)
