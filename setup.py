from setuptools import setup, find_packages

def get_description():
    return "MetaBind"

def get_scripts():
    return [
    ]


if __name__ == "__main__":
    setup(
        name="MetaBind",
        version="1.0.0",
        url="https://github.com/lucianlschan/MetaBind",
        description="MetaBind",
        long_description=get_description(),
        packages=find_packages(),
        package_data = {},
        scripts=get_scripts(),
        setup_requires=[],
        install_requires=["numpy", "scipy","rdkit","biopython","torch","dgl"],
        include_package_data=True,
        zip_safe = False,
        ext_modules=[],
        license="Apache License 2.0",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Scientific/Engineering :: Artificial Intelligence"
        ],
        keywords="Meta Learning",
        python_requires=">=3.7",
    )

