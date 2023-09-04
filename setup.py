
from distutils.core import setup
setup \
  ( name='mixjax'
  , version='0.1'
  , packages=['mixjax']
  , install_requires= \
    [ "numpy"
    , "scipy"
    , "jax"
    , "einops"
    ]
  )
