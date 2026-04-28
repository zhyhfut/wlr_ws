from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'wlr_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Zhaohy',
    maintainer_email='zhaohy@example.com',
    description='VMC + LQR balance controller for wheeled-legged robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'balance_node = wlr_controller.balance_node:main',
        ],
    },
)
