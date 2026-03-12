from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rh_cognitiv",
    version="0.0.0b0",
    author="rhodie",
    author_email="rhandref@gmail.com",
    description="Cognitive Skill-driven Orchestration Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rhodief/rh-cognitv",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'pydantic==2.12.5',
        'jsonpatch==1.33'
    ],
    package_data={
        "rh_cognitiv": ["py.typed"],
        "rh_cognitiv.execution_platform": ["py.typed"],
        "rh_cognitiv.orchestrators": ["py.typed"],
        "rh_cognitiv.cognitive": ["py.typed"],
    },
    include_package_data=True,
)
