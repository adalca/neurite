import setuptools

setuptools.setup(
    name='neurite',
    version='0.1',
    license='gpl-3.0',
    description='Neural Networks Toolbox for Medical Imaging',
    url='https://github.com/adalca/neurite',
    keywords=['imaging', 'cnn'],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'six',
        'numpy',
        'scipy',
        'tqdm',
        'matplotlib',
        'scikit-learn',
        'nibabel',
        'pystrum',
    ]
)
