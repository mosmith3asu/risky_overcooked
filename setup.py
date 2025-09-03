
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
        # "risky_overcooked_rl": [
        #     "static/**/*.pickle",
        #     "static/**/*.csv",
        #     "ppo/trained_example/*.pkl",
        #     "ppo/trained_example/*.json",
        #     "ppo/trained_example/*/.is_checkpoint",
        #     "ppo/trained_example/*/.tune_metadata",
        #     "ppo/trained_example/*/checkpoint-500",
        # ],
    },
    install_requires=[
        "dill",
        # "numpy<2.0.0",
        "numpy=1.26.3",
        "scipy=1.14.0",
        "tqdm=4.66.4",
        "gymnasium=0.28.1",
        "ipython=7.34.0",
        "pygame=2.5.2",
        "ipywidgets=8.1.2",
        "opencv-python=4.9.0.80",
        "flask=2.1.3",
        "flask-socketi=4.3.1",
        "numba=0.61.2"
    ],
    # removed overlapping dependencies
    extras_require={
        "harl": [
            "wandb",
            "GitPython",
            "memory_profiler",
            "sacred",
            "pymongo",
            "matplotlib=3.8.3",
            "requests",
            # "seaborn==0.9.0",
            # "ray[rllib]>=2.5.0",
            "protobuf",
            "torch==2.3.1+cu118"
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
# from setuptools import find_packages, setup
#
# setup(
#     name='risky_overcooked',
#     version='1.1.0',
#     url='https://github.com/ASU-RISE-Lab/risky_overcooked',
#     license='',
#     author='Mason O. Smith',
#     author_email='mosmith3@asu.edu',
#     description='Cooperative multi-agent environment based on Overcooked with risk decisions (adapted from Micah Carroll\'s work on overcooked_ai)',
#     packages=find_packages("src"),
#     # packages=['study_1', 'human_aware_rl', 'human_aware_rl.ppo', 'human_aware_rl.human', 'human_aware_rl.rllib',
#     #           'human_aware_rl.static', 'human_aware_rl.imitation', 'overcooked_demo', 'overcooked_demo.server',
#     #           'risky_overcooked_py', 'risky_overcooked_py.mdp', 'risky_overcooked_py.agents',
#     #           'risky_overcooked_py.planning', 'risky_overcooked_py.visualization', 'risky_overcooked_webserver',
#     #           'risky_overcooked_webserver.server'],
#     package_dir={'': 'src'},
#     package_data={
#             "risky_overcooked_py": [
#                 "data/layouts/*.layout",
#                 "data/planners/*.py",
#                 "data/human_data/*.pickle",
#                 "data/graphics/*.png",
#                 "data/graphics/*.json",
#                 "data/fonts/*.ttf",
#             ]
#     },
#     # install_requires=[
#     #     "dill",
#     #     # "numpy<2.0.0",
#     #     "numpy=1.26.3",
#     #     "scipy=1.14.0",
#     #     "tqdm=4.66.4",
#     #     "gymnasium=0.28.1",
#     #     "ipython=7.34.0",
#     #     "pygame=2.5.2",
#     #     "ipywidgets=8.1.2",
#     #     "opencv-python=4.9.0.80",
#     #     "flask=2.1.3",
#     #     "flask-socketi=4.3.1",
#     #     "numba=0.61.2"
#     # ],
#     # removed overlapping dependencies
#     # extras_require={
#     #     "harl": [
#     #         "wandb",
#     #         "GitPython",
#     #         "memory_profiler",
#     #         "sacred",
#     #         "pymongo",
#     #         "matplotlib=3.8.3",
#     #         "requests",
#     #         # "seaborn==0.9.0",
#     #         # "ray[rllib]>=2.5.0",
#     #         "protobuf",
#     #         "torch==2.3.1+cu118"
#     #         # "tensorflow>=2.14.0",
#     #     ]
#     # },
#  )
