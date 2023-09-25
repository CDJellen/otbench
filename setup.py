import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='otb',
    packages=setuptools.find_packages(exclude=['*tests*', '*notebooks*']),
    version='0.23.9.24',
    license='MIT',
    description='Consistent benchmarking for modeling optical turbulence strength.',
    author='Chris Jellen',
    author_email='cdjellen@gmail.com',
    url='https://github.com/cdjellen/ot-bench',
    long_description=long_description,
    long_description_content_type='text/markdown',
    download_url='https://github.com/cdjellen/ot-bench/tarball/main',
    keywords=['optics', 'turbulence', 'benchmark', 'otb', 'python3'],
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
