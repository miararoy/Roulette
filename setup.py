from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='roulette-ml',
    url='https://github.com/miararoy/roulette',
    author='Roy Miara',
    author_email='miararoy@gmail.com',
    # Needed to actually package something
    packages=[
        "roulette",
        "roulette.builder",
        "roulette.evaluation",
        "roulette.evaluation.plotting"
    ],
    # Needed for dependencies
    install_requires=[
        'numpy==1.15.2',
        'pandas==0.25.0',
        'scipy==1.10.0',
        'matplotlib==3.0.2',
        'seaborn==0.9.0',
        'scikit-learn==0.21.2',
        'joblib==0.13.2',
        'tqdm==4.31.1',
    ],
    # *strongly* suggested for sharing
    version='0.1.1',
    description='more than a metric',
    # We will also need a readme eventually (there will be a warning)
    long_description="TBD",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
