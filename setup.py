# Copyright 2019 Google LLC
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup script for qkerasV3."""


import setuptools

with open("README.md", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qkerasV3",
    version="1.0.0",
    author="qkerasV3 Team",
    author_email="qkerasV3-team@google.com",
    maintainer="Shan Li",
    maintainer_email="lishanok@google.com",
    packages=setuptools.find_packages(),
    scripts=[],
    url="",
    license="Apache v.2.0",
    description="Quantization package for Keras",
    long_description=long_description,
    install_requires=[
        "numpy",
        "scipy",
        "pyparser",
        "setuptools",
        "tensorflow-model-optimization",
        "networkx",
        "scikit-learn",
        "tqdm",
        "keras-tuner"
    ],
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=[
        "pytest",
    ],
)
