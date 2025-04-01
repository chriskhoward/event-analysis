from setuptools import setup, find_packages

setup(
    name="event_analysis",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'streamlit==1.32.0',
        'pandas==2.2.1',
        'plotly==5.19.0',
        'openpyxl==3.1.2',
        'scipy==1.12.0',
        'statsmodels==0.14.1',
        'fuzzywuzzy==0.18.0',
        'python-Levenshtein==0.25.0',
        'nameparser==1.1.1',
    ],
    entry_points={
        'console_scripts': [
            'event-analysis=event_analysis.app:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="An interactive event attendance analysis dashboard",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/event-analysis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
) 