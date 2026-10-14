# bstitcher-bcf

Bruker Esprit `.bcf` tile reader plugin for [bstitcher](../../README.md).

Licensed **GPL-3.0-or-later** because it builds on HyperSpy, RosettaSciIO and
eXSpy (all GPL-3.0). It registers itself via the `bstitcher.readers` entry
point (`bcf`); the Apache-2.0 core never imports it directly.
