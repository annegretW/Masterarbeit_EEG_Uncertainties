\page infrastructure Developer Infrastructure Notes

### Git
- Hosted at https://bitbucket.org/mituq/muq2

### Testing
- GTest for running c++ unit tests for each compile groups, sequential and MPI parallel
- RunAllNotebooks.sh for automatically running all python notebook examples

### CI
- See bitbucket-pipelines.yml in MUQ2 repo
- Coverage
  - Basic c++ unit tests and python notebooks test on push
  - Weekly extended runs mainly for different OS configs
  - Weekly installation instructions script run to ensure they it still leads to a running MUQ setup

### Website
- Hosted on bitbucket pages in the mituq.bitbucket.io repository.
- Accessible at https://mituq.bitbucket.io/
- When something is committed to this repository, bitbucket pipelines runs Jekyll and generate the html.
- (in progress) HTML example pages are generated from the documentation/scripts/ProcessExamples.py in the MUQ2 repository.   This script is run from the pipelines script in the MUQ2 repository when a new git tag is created.
- The bitbucket.io site is static, but the slack invitation button requires extra processing that cannot occur on a static site.  To overcome this,  a "cloud function" has been set up on the Google cloud platform to handle POST requests when the invitation modal and interact with the slack API.

### Doxygen
- Built via CI when release tag is defined on master repo, pushed as commit to website repo

### Docker (in progress)
- The docker images are built on knudsen.mit.edu.
- The build is started via a webhook when a new tag is created on the MUQ2 repository.
- Images are pushed to dockerhub https://hub.docker.com/r/mparno/muq

### Conda
- The MUQ2 conda image lives on conda-forge.
- The recipe for the MUQ2 conda image lives in the [muq-feedstock repository](https://github.com/conda-forge/muq-feedstock) on github.   
- Directions for updating the recipe manually can be found [here](https://conda-forge.org/docs/maintainer/updating_pkgs.html).
- Currently Matt is the only maintainer, but we should change that.
- In our bitbucket-pipelines.yml file, a fork of conda-forge's muq-feedstock repository is updated every time a new tag is created on the MUQ2 repository.   To update the image on conda-forge, we then need to go into github and create a pull request to merge our fork into conda-forge's repo.
