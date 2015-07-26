#!/bin/bash

find NLPGraph -name 'NLPGraph.build' -exec rm -rf {} \+
find NLPGraphTests -name 'NLPGraphTests.build' -exec rm -rf {} \+
find NLPGraph -iname '*cmake*' -not -name CMakeLists.txt -exec rm -rf {} \+
find NLPGraphTests -iname '*cmake*' -not -name CMakeLists.txt -exec rm -rf {} \+
find NLPGraph -iname 'Makefile' -exec rm -rf {} \+
find NLPGraphTests -iname 'Makefile' -exec rm -rf {} \+
rm Makefile
rm -rf CMakeFiles
rm *cmake*
