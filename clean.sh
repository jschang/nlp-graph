#!/bin/bash

for target in NLPGraph NLPGraphTests .; do
find . -name "$target.build" -exec rm -rvf {} \+
find $target -iname '*cmake*' -not -name CMakeLists.txt -exec rm -rvf {} \+
find $target -iname 'Makefile' -exec rm -rvf {} \+
find $target -name Debug -exec rm -rvf {} \+
find $target -iname *xcode* -exec rm -rvf {} \+
done;
