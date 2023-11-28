import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='otbench',
    packages=setuptools.find_packages(exclude=['*tests*', '*notebooks*']),
    version='0.23.11.28',
    license='MIT',
    description='Consistent benchmarks for evaluating optical turbulence strength models.',
    author='Chris Jellen',
    author_email='cdjellen@gmail.com',
    url='https://github.com/cdjellen/otbench',
    long_description=long_description,
    long_description_content_type='text/markdown',
    download_url='https://github.com/cdjellen/otbench/tarball/main',
    keywords=['optics', 'turbulence', 'benchmark', 'python3'],
    install_requires=['astral', 'pandas', 'requests', 'scikit-learn', 'xarray'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
