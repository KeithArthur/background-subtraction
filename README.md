# cvx-opt-project
### Setup
  * Download LSD data from link below
  * Install SPAMS from: [here](https://github.com/conda-forge/python-spams-feedstock) (if you use anaconda) or [here](http://spams-devel.gforge.inria.fr/downloads.html)
  * `pip install -r requirements.txt` if you don't have all the dependecies.

### Links
  * [Google Doc](https://docs.google.com/document/d/1J9UBF9qWj5F0dgwaR13iziSsxwNf4S1EjGhgPV-Wc1g/edit)
  * [Low Rank Sparsity Decomposition code and data](http://www.ee.oulu.fi/~xliu/research/lsd/LSD.html)
  * [RPCA_GD Code](http://www.yixinyang.org/code/RPCA_GD.zip)

### General recommendations
  * Work in a separate branch then open a pull request on github so we can easily collaborate without conflicts.
  * Only commit code to git. Large data should be shared some other way
  * Use descriptive variable and function names.
  * Prepend function names with _ if they should not be called directly by other parts of the code.
  * Write short functions.
  * Append variable names with symbol used in literature if you feel it would help. For example `video_frames_D` or `background_L`.
  * Write some tests if you want :)
