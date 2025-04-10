
from setuptools import find_packages, setup

with open("README.md", "r", encoding="UTF8") as fh:
    long_description = fh.read()

setup(
    name="risky_overcooked",
    version="1.1.0",
    description="Cooperative multi-agent environment based on Overcooked with risk decisions (adapted from Micah Carroll's work on overcooked_ai)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mason O. Smith",
    author_email="mosmith3@asu.edu",
    url="https://github.com/mosmith3asu/risky_overcooked",
    # download_url="https://github.com/HumanCompatibleAI/overcooked_ai/archive/refs/tags/1.1.0.tar.gz",
    packages=find_packages("src"),
    keywords=["Overcooked", "AI", "Reinforcement Learning"],
    package_dir={"": "src"},
    package_data={
        "risky_overcooked_py": [
            "data/layouts/*.layout",
            "data/planners/*.py",
            "data/human_data/*.pickle",
            "data/graphics/*.png",
            "data/graphics/*.json",
            "data/fonts/*.ttf",
        ],
        "risky_overcooked_rl": [
            "static/**/*.pickle",
            "static/**/*.csv",
            "ppo/trained_example/*.pkl",
            "ppo/trained_example/*.json",
            "ppo/trained_example/*/.is_checkpoint",
            "ppo/trained_example/*/.tune_metadata",
            "ppo/trained_example/*/checkpoint-500",
        ],
    },
    install_requires=[
        "dill",
        # "numpy<2.0.0",
        "numpy=1.24.3",
        "scipy",
        "tqdm",
        "gymnasium",
        "ipython",
        "pygame",
        "ipywidgets",
        "opencv-python",
        "flask",
        "flask-socketio",
    ],
    # removed overlapping dependencies
    extras_require={
        "harl": [
            "wandb",
            "GitPython",
            "memory_profiler",
            "sacred",
            "pymongo",
            "matplotlib",
            "requests",
            # "seaborn==0.9.0",
            # "ray[rllib]>=2.5.0",
            "protobuf",
            # "tensorflow>=2.14.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "overcooked-demo-up = overcooked_demo:start_server",
            "overcooked-demo-move = overcooked_demo:move_agent",
        ]
    },
)