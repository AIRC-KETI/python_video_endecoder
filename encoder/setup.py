#!/usr/bin/env python3
# encoding: utf-8

from distutils.core import setup, Extension
import subprocess
import numpy as np

def pkgconfig(*packages, **kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    for token in subprocess.getoutput("pkg-config --libs --cflags %s" % ' '.join(packages)).split():
        if token[:2] in flag_map:
            kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
        else: # throw others to extra_link_args
            kw.setdefault('extra_link_args', []).append(token)
    for k, v in kw.items(): # remove duplicated
        kw[k] = list(set(v))
    return kw

options = pkgconfig('libavutil', 'libavcodec', 'libavformat', 'libswscale')
options['include_dirs'].append(np.get_include())

videoencoder_module = Extension('videoencoder', ['video_encoder.cpp'], **options)

setup(name='videoencoder',
      version='0.1.0',
      description='Video encoder module.',
ext_modules=[videoencoder_module])