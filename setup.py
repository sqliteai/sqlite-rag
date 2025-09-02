import setuptools
import toml

with open("pyproject.toml", "r") as f:
    pyproject = toml.load(f)

project = pyproject["project"]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name=project["name"],
    version=project["version"],
    description=project.get("description", ""),
    author=project["authors"][0]["name"] if project.get("authors") else "",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=project["urls"]["Homepage"],
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=project.get("requires-python", ">=3.10"),
    classifiers=project.get("classifiers", []),
)
