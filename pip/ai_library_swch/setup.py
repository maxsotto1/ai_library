import io
import os
import re
from setuptools import setup, find_packages


def read(fname):
    here = os.path.abspath(os.path.dirname(__file__))
    try:
        with io.open(os.path.join(here, fname), encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def parse_requirements(path="ai_library/requirements.txt"):
    reqs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    reqs.append(line)
    except FileNotFoundError:
        pass
    return reqs


setup(
    packages=find_packages(),
    install_requires=parse_requirements(),
)
