from setuptools import setup

setup(
    name='risky_overcooked',
    version='1.1.0',
    packages=['study_1', 'human_aware_rl', 'human_aware_rl.ppo', 'human_aware_rl.human', 'human_aware_rl.rllib',
              'human_aware_rl.static', 'human_aware_rl.imitation', 'overcooked_demo', 'overcooked_demo.server',
              'risky_overcooked_py', 'risky_overcooked_py.mdp', 'risky_overcooked_py.agents',
              'risky_overcooked_py.planning', 'risky_overcooked_py.visualization', 'risky_overcooked_webserver',
              'risky_overcooked_webserver.server'],
    package_dir={'': 'src'},
    url='https://github.com/ASU-RISE-Lab/risky_overcooked',
    license='',
    author='Mason O. Smith',
    author_email='mosmith3@asu.edu',
    description='Cooperative multi-agent environment based on Overcooked with risk decisions (adapted from Micah Carroll\'s work on overcooked_ai)'
)
