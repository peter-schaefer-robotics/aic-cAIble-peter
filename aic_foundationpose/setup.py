from setuptools import find_packages, setup
import os
from glob import glob

package_name = "aic_foundationpose"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Richard",
    maintainer_email="ge26roj@mytum.de",
    description="FoundationPose 6-DoF pose estimation for NIC Card",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "foundationpose_node = aic_foundationpose.foundationpose_node:main",
            "convert_nic_mesh = aic_foundationpose.convert_mesh:main",
        ],
    },
)
