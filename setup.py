from setuptools import setup, find_packages

setup(name='coffail_utils',
      version='1.0.0',
      description='A package providing components for working with the COFFAIL dataset',
      author='Alex Mitrevski',
      author_email='alemitr@chalmers.se',
      keywords='robotics robot_skill_learning robot_failures',
      packages=find_packages(exclude=['contrib', 'docs', 'tests'])
)
